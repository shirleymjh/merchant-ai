from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.config import Settings
from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    DataSnapshotContract,
    EntitySet,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_execution_policy import GroundedExecutionMode
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
    GroundedContextWorkspaceError,
    validated_grounded_query_artifact_roots,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedSelectedFieldBinding,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeSession,
    verified_query_artifact_integrity_fingerprint,
    verified_query_artifact_integrity_valid,
)


class PublicationExecutor:
    def __init__(
        self,
        *,
        fail_publication: bool = False,
        row_count: int = 1,
        preview_rows: int | None = None,
    ) -> None:
        self.fail_publication = fail_publication
        self.row_count = row_count
        self.publish_calls = 0
        if preview_rows is not None:
            self.settings = SimpleNamespace(
                context_artifact_inline_max_rows=preview_rows
            )
        self.on_publish: Any = None

    def execute_contract(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        pack: PlanningAssetPack,
        question: str,
        **kwargs: Any,
    ) -> AgentRunResult:
        del merchant_id, contract, plan, pack, question
        task_id = "task-1"
        pending_id = "pending-1"
        private_receipt = {
            "pendingArtifactId": pending_id,
            "publicationRoot": "/server/private/artifacts",
            "stagingRoot": "/server/private/staging",
            "identity": {
                "taskFingerprint": hashlib.sha256(
                    task_id.encode("utf-8")
                ).hexdigest(),
            },
        }
        bundle = QueryBundle(
            sql="SELECT 1 AS value",
            rows=[{"value": index + 1} for index in range(self.row_count)],
            original_row_count=self.row_count,
            result_coverage="ALL_ROWS",
            data_snapshot=DataSnapshotContract(
                datasource_fingerprint="test-datasource",
                semantic_activation_fingerprint=str(
                    kwargs[
                        "expected_semantic_activation_fingerprint"
                    ]
                ),
                unsupported_reason="TEST_EPOCH_UNAVAILABLE",
            ),
            runtime_events=[
                {
                    "event": "grounded_data_engine.executed",
                    "taskId": task_id,
                    "_serverPrivatePendingResultArtifact": private_receipt,
                }
            ],
        )
        assert kwargs["execution_generation"] == 1
        assert kwargs["execution_attempt_id"] == "attempt-1"
        return AgentRunResult(
            query_bundles=[bundle.model_copy(deep=True)],
            merged_query_bundle=bundle.model_copy(deep=True),
        )

    def publish_pending_result_artifact(
        self,
        pending_receipt: dict[str, Any],
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self.publish_calls += 1
        if callable(self.on_publish):
            self.on_publish()
        if self.fail_publication:
            raise RuntimeError("simulated publication failure")
        digest = "a" * 64
        return {
            "artifactFingerprint": digest,
            "pendingArtifactId": pending_receipt["pendingArtifactId"],
            "manifestRelativePath": "query_results/result.manifest.json",
            "manifestRef": "merchant://artifact/query_results/result.manifest.json",
            "rowsRef": "merchant://artifact/query_results/result_rows.json",
            "sqlRef": "merchant://artifact/query_results/result.sql",
            "queryManifestSha256": digest,
            "rowsSha256": digest,
            "sqlSha256": digest,
            "verifiedEvidenceSha256": digest,
            "resultCoverage": "ALL_ROWS",
        }


class FixedVerifier:
    def __init__(self, passed: bool) -> None:
        self.passed = passed

    def verify(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
    ) -> VerifiedEvidence:
        del question, plan, run_result
        return VerifiedEvidence(passed=self.passed)


class PublicationSemanticAuthority:
    @staticmethod
    def all_topic_names() -> list[str]:
        return ["topic_a"]

    @staticmethod
    def semantic_source_hash(topics: list[str]) -> str:
        return hashlib.sha256(
            "|".join(sorted(set(topics))).encode("utf-8")
        ).hexdigest()


def _kernel_and_session(
    *,
    verifier_passed: bool,
    fail_publication: bool = False,
    row_count: int = 1,
    preview_rows: int | None = None,
) -> tuple[GroundedRuntimeKernel, GroundedRuntimeSession, PublicationExecutor]:
    executor = PublicationExecutor(
        fail_publication=fail_publication,
        row_count=row_count,
        preview_rows=preview_rows,
    )
    kernel = GroundedRuntimeKernel(
        PublicationSemanticAuthority(),
        keyword_service=object(),
        topic_router=object(),
        executor=executor,
        verifier=FixedVerifier(verifier_passed),
    )
    session = kernel.new_session(
        "question",
        "merchant-1",
        session_id="session-1",
    )
    session.workspace_topics = ["topic_a"]
    session.active_generation = 1
    session.active_attempt_id = "attempt-1"
    session.active_execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
    session.active_contract = GroundedQueryContract(
        question="question",
        topics=["topic_a"],
        status="READY",
        query_shape="SCALAR",
    )
    session.active_plan = QueryPlan()
    session.active_pack = PlanningAssetPack()
    session.active_preparation = SimpleNamespace(
        executable=True,
        plan=QueryPlan(),
        asset_pack_fingerprint="semantic-activation-1",
    )
    return kernel, session, executor


def _execute(kernel: GroundedRuntimeKernel, session: GroundedRuntimeSession) -> None:
    kernel.execute_active(
        session,
        run_id="run-1",
        artifact_root="/server/private/artifacts",
        context_owner_fingerprint="server-owner",
        goal_contract_fingerprint="a" * 64,
    )


def test_verifier_failure_never_publishes_or_enters_verified_ledger() -> None:
    kernel, session, executor = _kernel_and_session(verifier_passed=False)
    _execute(kernel, session)

    event = session.run_result.merged_query_bundle.runtime_events[0]
    assert "_serverPrivatePendingResultArtifact" not in event
    assert "resultArtifact" not in event

    verified = kernel.verify_active(session)

    assert verified.passed is False
    assert executor.publish_calls == 0
    assert session.verified_query_ledger == []
    assert session.pending_query_publications[0].status == "VERIFICATION_FAILED"
    assert session.pending_query_publications[0].receipt == {}
    assert session.publication_authority_run_result is None


def test_serial_publication_failure_blocks_verified_ledger_commit() -> None:
    kernel, session, executor = _kernel_and_session(
        verifier_passed=True,
        fail_publication=True,
    )
    _execute(kernel, session)

    verified = kernel.verify_active(session)

    assert verified.passed is False
    assert verified.blocking_gaps[0].code == (
        "QUERY_RESULT_ARTIFACT_PUBLICATION_FAILED"
    )
    assert executor.publish_calls == 1
    assert session.verified_query_ledger == []
    assert session.pending_query_publications[0].status == "PUBLICATION_FAILED"
    assert session.pending_query_publications[0].receipt == {}
    assert session.publication_authority_run_result is None


def test_serial_verified_commit_exposes_only_opaque_published_receipt() -> None:
    kernel, session, executor = _kernel_and_session(verifier_passed=True)
    _execute(kernel, session)

    verified = kernel.verify_active(session)

    assert verified.passed is True
    assert executor.publish_calls == 1
    artifact = session.verified_query_ledger[0]
    assert artifact.publication_status == "PUBLISHED"
    assert artifact.result_artifact_receipts[0]["queryManifestSha256"]
    bundle = artifact.run_result.merged_query_bundle
    assert bundle.offloaded_files == []
    assert bundle.runtime_events[0]["resultArtifact"]["manifestRef"].startswith(
        "merchant://"
    )
    assert all(
        not str(value).startswith("/")
        for receipt in artifact.result_artifact_receipts
        for value in receipt.values()
    )
    assert session.publication_authority_run_result is None
    assert session.publication_authority_fingerprint == ""
    assert session.pending_query_publications[0].receipt == {}


def test_physical_publication_runs_outside_the_kernel_lock() -> None:
    kernel, session, executor = _kernel_and_session(verifier_passed=True)
    lock_observations: list[bool] = []

    def observe_lock_from_another_thread() -> None:
        def probe() -> bool:
            acquired = kernel._lock.acquire(blocking=False)
            if acquired:
                kernel._lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as pool:
            lock_observations.append(pool.submit(probe).result())

    executor.on_publish = observe_lock_from_another_thread
    _execute(kernel, session)

    assert kernel.verify_active(session).passed is True
    assert lock_observations == [True]


def test_execute_keeps_one_private_full_result_and_public_bounded_projection() -> None:
    kernel, session, _executor = _kernel_and_session(
        verifier_passed=True,
        row_count=3,
        preview_rows=1,
    )

    public_result = kernel.execute_active(
        session,
        run_id="run-1",
        artifact_root="/server/private/artifacts",
        context_owner_fingerprint="server-owner",
        goal_contract_fingerprint="a" * 64,
    )

    assert public_result is session.run_result
    assert public_result.merged_query_bundle.rows == [{"value": 1}]
    assert public_result.merged_query_bundle.result_coverage == "PREVIEW"
    assert session.publication_authority_run_result is not None
    assert session.publication_authority_run_result is not public_result
    assert session.publication_authority_run_result.merged_query_bundle.rows == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]

    assert kernel.verify_active(session).passed is True

    artifact = session.verified_query_ledger[0]
    assert artifact.run_result.merged_query_bundle.rows == [
        {"value": 1},
        {"value": 2},
        {"value": 3},
    ]
    assert session.run_result.merged_query_bundle.rows == [{"value": 1}]
    assert session.publication_authority_run_result is None


def test_public_run_result_mutation_cannot_change_publication_authority() -> None:
    kernel, session, _executor = _kernel_and_session(verifier_passed=True)
    _execute(kernel, session)
    session.run_result.merged_query_bundle.rows[0]["value"] = 999
    session.run_result.query_bundles[0].rows[0]["value"] = 999

    verified = kernel.verify_active(session)

    assert verified.passed is True
    artifact = session.verified_query_ledger[0]
    assert artifact.run_result.merged_query_bundle.rows == [{"value": 1}]
    assert artifact.run_result.query_bundles[0].rows == [{"value": 1}]


def test_nested_run_result_mutation_invalidates_public_ledger_seal() -> None:
    kernel, session, _executor = _kernel_and_session(verifier_passed=True)
    _execute(kernel, session)
    assert kernel.verify_active(session).passed is True
    artifact = session.verified_query_ledger[0]

    assert verified_query_artifact_integrity_valid(artifact) is True
    assert artifact.ledger_fingerprint == (
        verified_query_artifact_integrity_fingerprint(artifact)
    )

    artifact.run_result.merged_query_bundle.rows[0]["value"] = 999

    assert verified_query_artifact_integrity_valid(artifact) is False


def test_deferred_branch_stays_invisible_until_adoption_gate() -> None:
    kernel, parent, executor = _kernel_and_session(verifier_passed=True)
    parent.active_generation = 0
    parent.active_attempt_id = ""
    parent.active_contract = None
    parent.active_plan = None
    parent.active_pack = None
    parent.active_preparation = None
    branch = kernel.fork_query_branch(parent, "branch-1")
    branch.active_generation = 1
    branch.active_attempt_id = "attempt-1"
    branch.active_execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
    branch.active_contract = GroundedQueryContract(
        question="question",
        status="READY",
        query_shape="SCALAR",
    )
    branch.active_plan = QueryPlan()
    branch.active_pack = PlanningAssetPack()
    branch.active_preparation = SimpleNamespace(
        executable=True,
        plan=QueryPlan(),
        asset_pack_fingerprint="semantic-activation-1",
    )

    _execute(kernel, branch)
    verified = kernel.verify_active(branch)

    assert verified.passed is True
    assert executor.publish_calls == 0
    assert parent.verified_query_ledger == []
    assert branch.verified_query_ledger[0].publication_status == "PENDING"
    assert branch.publication_authority_run_result is None
    assert branch.pending_query_publications[0].receipt
    assert "resultArtifact" not in (
        branch.run_result.merged_query_bundle.runtime_events[0]
    )

    adopted = kernel.adopt_verified_branches(parent, [branch])

    assert executor.publish_calls == 1
    assert [item.artifact_id for item in parent.verified_query_ledger] == [
        item.artifact_id for item in adopted
    ]
    assert adopted[0].publication_status == "PUBLISHED"
    assert branch.publication_authority_run_result is None
    assert branch.publication_authority_fingerprint == ""
    assert branch.pending_query_publications[0].receipt == {}
    assert branch.verified_query_ledger[0].run_result is not (
        parent.verified_query_ledger[0].run_result
    )


def test_in_place_branch_ledger_mutation_fails_before_publication() -> None:
    kernel, parent, executor = _kernel_and_session(verifier_passed=True)
    parent.active_generation = 0
    parent.active_attempt_id = ""
    parent.active_contract = None
    parent.active_plan = None
    parent.active_pack = None
    parent.active_preparation = None
    branch = kernel.fork_query_branch(parent, "branch-1")
    branch.active_generation = 1
    branch.active_attempt_id = "attempt-1"
    branch.active_execution_mode = GroundedExecutionMode.DETERMINISTIC_METRIC
    branch.active_contract = GroundedQueryContract(
        question="question",
        status="READY",
        query_shape="SCALAR",
    )
    branch.active_plan = QueryPlan()
    branch.active_pack = PlanningAssetPack()
    branch.active_preparation = SimpleNamespace(
        executable=True,
        plan=QueryPlan(),
        asset_pack_fingerprint="semantic-activation-1",
    )
    _execute(kernel, branch)
    assert kernel.verify_active(branch).passed is True
    branch.verified_query_ledger[0].output_columns.append("forged")

    with pytest.raises(RuntimeError) as raised:
        kernel.adopt_verified_branches(parent, [branch])
    assert "VERIFIED_BRANCH_ARTIFACT_PUBLICATION_FAILED" in str(raised.value)

    assert executor.publish_calls == 0
    assert parent.verified_query_ledger == []


def test_staging_root_symlink_is_rejected_without_touching_target(
    tmp_path: Path,
) -> None:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    workspace = GroundedContextWorkspace.open(
        settings,
        thread_id="thread-1",
        run_id="run-1",
        merchant_id="merchant-1",
        access_role="merchant_analyst",
        user_scope={},
        question="question",
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace.staging_root.rmdir()
    workspace.staging_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(GroundedContextWorkspaceError) as raised:
        validated_grounded_query_artifact_roots(
            settings.resolved_workspace_path,
            workspace.artifacts_root,
        )
    assert "QUERY_RESULT_PRECREATED_ROOT_REQUIRED" in str(raised.value)

    assert list(outside.iterdir()) == []


def _entity_artifact_session(
    coverage: str,
    *,
    truncated: bool,
    row_count: int,
) -> tuple[GroundedRuntimeKernel, GroundedRuntimeSession, str]:
    kernel, session, _executor = _kernel_and_session(verifier_passed=True)
    output = "entity_id"
    rows = [{output: "entity-%d" % index} for index in range(row_count)]
    contract = GroundedQueryContract(
        question="entities",
        status="READY",
        query_shape="DETAIL",
        selected_fields=[
            GroundedSelectedFieldBinding(
                semantic_ref_id="semantic:entity:id",
                topic="entity",
                table="entity_table",
                column=output,
                output_alias=output,
                entity_identity="entity:test",
            )
        ],
    )
    plan = QueryPlan()
    run_result = AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="task-1",
                entity_set=EntitySet(
                    task_id="task-1",
                    join_key=output,
                    values=[row[output] for row in rows],
                    column_values={
                        output: [row[output] for row in rows]
                    },
                    truncated=False,
                    source_row_count=row_count,
                ),
            )
        ],
        merged_query_bundle=QueryBundle(
            rows=rows,
            original_row_count=(0 if coverage == "PREVIEW" else row_count),
            result_coverage=coverage,
            is_truncated=truncated,
        ),
    )
    session.active_contract = contract
    session.active_plan = plan
    session.verified_query_ledger = []
    artifact = kernel._record_verified_query_artifact(
        session,
        generation=1,
        plan=plan,
        run_result=run_result,
        verified=VerifiedEvidence(passed=True),
    )
    assert artifact.sealed_entity_values_truncated is truncated
    return kernel, session, output


def test_preview_detail_population_cannot_publish_entity_set() -> None:
    # The deterministic detail executor fetches row 101 as a sentinel and
    # exposes only 100 rows with PREVIEW coverage.
    kernel, session, output = _entity_artifact_session(
        "PREVIEW",
        truncated=True,
        row_count=100,
    )

    with pytest.raises(RuntimeError) as raised:
        kernel.publish_verified_entity_set(
            session,
            session.verified_query_ledger[0].artifact_id,
            output,
        )
    assert "VERIFIED_ENTITY_SET_INCOMPLETE_COVERAGE:PREVIEW" in str(
        raised.value
    )


@pytest.mark.parametrize("coverage", ["ALL_ROWS", "TOP_N"])
def test_complete_or_bounded_topn_population_can_publish_entity_set(
    coverage: str,
) -> None:
    kernel, session, output = _entity_artifact_session(
        coverage,
        truncated=False,
        row_count=3,
    )

    entity_set = kernel.publish_verified_entity_set(
        session,
        session.verified_query_ledger[0].artifact_id,
        output,
    )

    assert entity_set.value_count == 3
    assert entity_set.truncated is False
