from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace
from typing import Any

import pytest

from merchant_ai.models import (
    AgentRunResult,
    DataSnapshotContract,
    PlanningAssetPack,
    QueryBundle,
    QueryPlan,
    VerifiedEvidence,
)
from merchant_ai.services.grounded_context_compaction import (
    build_grounded_recovery_payload,
)
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionGraphProposal,
    GroundedExecutionNodeSpec,
    build_grounded_execution_graph_receipt,
)
from merchant_ai.services.grounded_execution_policy import (
    GroundedExecutionMode,
)
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
)
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
)
from merchant_ai.services.grounded_semantic_activation import (
    semantic_activation_seal_valid,
)


class MutableSemanticAuthority:
    def __init__(self, topics: dict[str, str]) -> None:
        self.sources = dict(topics)
        self.barrier: Barrier | None = None

    def all_topic_names(self) -> list[str]:
        return sorted(self.sources)

    def semantic_source_hash(self, topics: list[str]) -> str:
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        payload = {
            topic: self.sources[topic]
            for topic in sorted(set(topics))
            if topic in self.sources
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class CapturingExecutor:
    def __init__(self, *, snapshot_override: str = "") -> None:
        self.snapshot_override = snapshot_override
        self.captured_activation = ""
        self.execute_calls = 0
        self.seen_snapshot: DataSnapshotContract | None = None
        self.settings = SimpleNamespace(
            context_artifact_inline_max_rows=10
        )

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        self.captured_activation = semantic_activation_fingerprint
        return DataSnapshotContract(
            datasource_fingerprint="d" * 64,
            datasource_environment="test",
            semantic_activation_fingerprint=(
                self.snapshot_override
                or semantic_activation_fingerprint
            ),
            cache_generation="generation-1",
            unsupported_reason="TEST_OBSERVED_EPOCH_UNAVAILABLE",
        )

    def execute_contract(
        self,
        merchant_id: str,
        contract: GroundedQueryContract,
        plan: QueryPlan,
        asset_pack: PlanningAssetPack,
        question: str,
        **kwargs: Any,
    ) -> AgentRunResult:
        del merchant_id, contract, plan, asset_pack, question
        self.execute_calls += 1
        snapshot = kwargs.get("data_snapshot_contract")
        if snapshot is None:
            snapshot = self.capture_data_snapshot(
                str(
                    kwargs.get(
                        "expected_semantic_activation_fingerprint"
                    )
                    or ""
                )
            )
        assert isinstance(snapshot, DataSnapshotContract)
        self.seen_snapshot = snapshot.model_copy(deep=True)
        return AgentRunResult(
            merged_query_bundle=QueryBundle(
                rows=[{"value": 1}],
                original_row_count=1,
                data_snapshot=snapshot.model_copy(deep=True),
            )
        )


class PassingVerifier:
    @staticmethod
    def verify(
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult,
    ) -> VerifiedEvidence:
        del question, plan, run_result
        return VerifiedEvidence(passed=True)


def _activated_runtime(
    authority: MutableSemanticAuthority,
    executor: CapturingExecutor,
) -> tuple[GroundedRuntimeKernel, Any]:
    kernel = GroundedRuntimeKernel(
        authority,
        keyword_service=object(),
        topic_router=object(),
        executor=executor,
        verifier=PassingVerifier(),
    )
    session = kernel.new_session(
        "neutral objective",
        "principal-1",
        session_id="session-1",
    )
    session.workspace_topics = ["topic_a"]
    session.active_generation = 1
    session.active_attempt_id = "attempt-1"
    session.active_execution_mode = (
        GroundedExecutionMode.DETERMINISTIC_METRIC
    )
    session.active_contract = GroundedQueryContract(
        question=session.question,
        topics=["topic_a"],
        status="READY",
        query_shape="SCALAR",
    )
    session.active_plan = QueryPlan()
    session.active_pack = PlanningAssetPack()
    session.active_preparation = SimpleNamespace(
        executable=True,
        plan=QueryPlan(),
        asset_pack_fingerprint="f" * 64,
    )
    return kernel, session


def test_semantic_activation_source_change_is_typed_stale() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-1"})
    kernel, session = _activated_runtime(
        authority,
        CapturingExecutor(),
    )
    seal = kernel.seal_semantic_activation(
        session,
        ["topic_a"],
    )
    assert seal is not None
    assert semantic_activation_seal_valid(seal)

    authority.sources["topic_a"] = "source-2"

    with pytest.raises(RuntimeError) as raised:
        kernel.revalidate_semantic_activation(session)
    assert "SEMANTIC_ACTIVATION_STALE" in str(raised.value)


def test_topic_expansion_reseals_only_before_execution() -> None:
    authority = MutableSemanticAuthority(
        {
            "topic_a": "source-a",
            "topic_b": "source-b",
            "topic_c": "source-c",
        }
    )
    kernel, session = _activated_runtime(
        authority,
        CapturingExecutor(),
    )
    first = kernel.seal_semantic_activation(session, ["topic_a"])
    expanded = kernel.seal_semantic_activation(
        session,
        ["topic_a", "topic_b"],
        allow_topic_expansion=True,
    )

    assert first is not None and expanded is not None
    assert expanded.version == first.version + 1
    assert expanded.exact_topics == ["topic_a", "topic_b"]
    assert expanded.semantic_activation_fingerprint != (
        first.semantic_activation_fingerprint
    )

    session.semantic_activation_execution_started = True
    with pytest.raises(RuntimeError) as raised:
        kernel.seal_semantic_activation(
            session,
            ["topic_c"],
            allow_topic_expansion=True,
        )
    assert (
        "SEMANTIC_ACTIVATION_TOPIC_EXPANSION_AFTER_EXECUTION_FORBIDDEN"
        in str(raised.value)
    )


def test_concurrent_initial_seal_uses_one_cas_identity() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    kernel, session = _activated_runtime(
        authority,
        CapturingExecutor(),
    )
    authority.barrier = Barrier(2)

    with ThreadPoolExecutor(max_workers=2) as pool:
        seals = list(
            pool.map(
                lambda _index: kernel.seal_semantic_activation(
                    session,
                    ["topic_a"],
                ),
                range(2),
            )
        )

    assert all(item is not None for item in seals)
    assert {item.seal_fingerprint for item in seals if item} == {
        session.semantic_activation_seal.seal_fingerprint
    }
    assert session.semantic_activation_seal.version == 1


def test_branch_inherits_parent_activation_without_narrowing_digest() -> None:
    authority = MutableSemanticAuthority(
        {"topic_a": "source-a", "topic_b": "source-b"}
    )
    kernel, parent = _activated_runtime(
        authority,
        CapturingExecutor(),
    )
    parent.workspace_topics = ["topic_a", "topic_b"]
    parent_seal = kernel.seal_semantic_activation(
        parent,
        parent.workspace_topics,
    )

    branch = kernel.fork_query_branch(
        parent,
        "node_a",
        workspace_topics=["topic_a"],
    )

    assert parent_seal is not None
    assert branch.semantic_activation_seal is not None
    assert (
        branch.semantic_activation_seal.seal_fingerprint
        == parent_seal.seal_fingerprint
    )
    assert branch.semantic_activation_seal.exact_topics == [
        "topic_a",
        "topic_b",
    ]


def test_multi_node_execution_and_adoption_preserve_one_activation() -> None:
    authority = MutableSemanticAuthority(
        {"topic_a": "source-a", "topic_b": "source-b"}
    )
    executor = CapturingExecutor()
    kernel = GroundedRuntimeKernel(
        authority,
        keyword_service=object(),
        topic_router=object(),
        executor=executor,
        verifier=PassingVerifier(),
    )
    parent = kernel.new_session(
        "multi node objective",
        "principal-1",
        session_id="parent-session",
    )
    parent.workspace_topics = ["topic_a", "topic_b"]
    parent_seal = kernel.seal_semantic_activation(
        parent,
        parent.workspace_topics,
    )
    branches = [
        kernel.fork_query_branch(
            parent,
            "node_a",
            workspace_topics=["topic_a"],
        ),
        kernel.fork_query_branch(
            parent,
            "node_b",
            workspace_topics=["topic_b"],
        ),
    ]
    for index, branch in enumerate(branches, start=1):
        branch.active_generation = 1
        branch.active_attempt_id = "attempt-%d" % index
        branch.active_execution_mode = (
            GroundedExecutionMode.DETERMINISTIC_METRIC
        )
        branch.active_contract = GroundedQueryContract(
            question=branch.question,
            topics=list(branch.workspace_topics),
            status="READY",
            query_shape="SCALAR",
        )
        branch.active_plan = QueryPlan()
        branch.active_pack = PlanningAssetPack()
        branch.active_preparation = SimpleNamespace(
            executable=True,
            plan=QueryPlan(),
            asset_pack_fingerprint="f" * 64,
        )

    def execute_and_verify(branch: Any) -> None:
        kernel.execute_active(branch)
        assert kernel.verify_active(branch).passed

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute_and_verify, branches))
    adopted = kernel.adopt_verified_branches(parent, branches)

    assert parent_seal is not None
    assert len(adopted) == 2
    assert {
        item.semantic_activation_fingerprint for item in adopted
    } == {parent_seal.semantic_activation_fingerprint}
    assert {
        item.semantic_activation_seal_fingerprint for item in adopted
    } == {parent_seal.seal_fingerprint}
    assert parent.semantic_activation_execution_started is True


def test_execute_captures_snapshot_from_semantic_seal_not_pack_hash() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    executor = CapturingExecutor()
    kernel, session = _activated_runtime(authority, executor)

    result = kernel.execute_active(session)
    verified = kernel.verify_active(session)

    seal = session.semantic_activation_seal
    assert seal is not None
    assert executor.captured_activation == (
        seal.semantic_activation_fingerprint
    )
    assert executor.captured_activation != "f" * 64
    assert result.merged_query_bundle.data_snapshot.semantic_activation_fingerprint == (
        seal.semantic_activation_fingerprint
    )
    assert verified.passed
    artifact = session.verified_query_ledger[-1]
    assert artifact.semantic_activation_fingerprint == (
        seal.semantic_activation_fingerprint
    )
    assert artifact.semantic_activation_seal_fingerprint == (
        seal.seal_fingerprint
    )
    assert artifact.semantic_activation_topics == ["topic_a"]


def test_execute_fails_before_doris_when_semantic_source_changed() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    executor = CapturingExecutor()
    kernel, session = _activated_runtime(authority, executor)
    kernel.seal_semantic_activation(session, ["topic_a"])
    authority.sources["topic_a"] = "source-b"

    with pytest.raises(RuntimeError) as raised:
        kernel.execute_active(session)

    assert "SEMANTIC_ACTIVATION_STALE" in str(raised.value)
    assert executor.execute_calls == 0


def test_invalid_goal_identity_is_rejected_before_executor_or_snapshot() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    executor = CapturingExecutor()
    kernel, session = _activated_runtime(authority, executor)

    with pytest.raises(RuntimeError) as raised:
        kernel.execute_active(
            session,
            artifact_root="/trusted/query-artifacts",
            context_owner_fingerprint="context-owner",
            goal_contract_fingerprint="not-a-content-fingerprint",
        )

    assert str(raised.value) == (
        "QUERY_EXECUTION_GOAL_CONTRACT_FINGERPRINT_INVALID"
    )
    assert executor.execute_calls == 0
    assert executor.captured_activation == ""


def test_snapshot_activation_mismatch_fails_closed() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    executor = CapturingExecutor(snapshot_override="e" * 64)
    kernel, session = _activated_runtime(authority, executor)

    with pytest.raises(RuntimeError) as raised:
        kernel.execute_active(session)

    assert "DATA_SNAPSHOT_SEMANTIC_ACTIVATION_MISMATCH" in str(
        raised.value
    )
    # The executor owns snapshot capture after SQL/ACL validation. The
    # mismatch is still rejected before its business-query path can return.
    assert executor.execute_calls == 1


def test_graph_topology_fingerprint_is_separate_from_activation() -> None:
    proposal = GroundedExecutionGraphProposal(
        base_version=0,
        goal_contract_fingerprint="g" * 64,
        discovery_snapshot_fingerprint="d" * 64,
        nodes=[
            GroundedExecutionNodeSpec(
                client_key="node_a",
                objective="objective",
                goal_ids=["goal_a"],
                topic_scope=["topic_a"],
                evidence_ref_ids=["semantic:topic_a:asset_a"],
            )
        ],
    )
    first = build_grounded_execution_graph_receipt(
        proposal,
        version=1,
        semantic_activation_fingerprint="a" * 64,
        semantic_activation_seal_fingerprint="b" * 64,
        semantic_activation_topics=["topic_a"],
    )
    second = build_grounded_execution_graph_receipt(
        proposal,
        version=1,
        semantic_activation_fingerprint="c" * 64,
        semantic_activation_seal_fingerprint="d" * 64,
        semantic_activation_topics=["topic_a"],
    )

    assert first.fingerprint == second.fingerprint
    assert first.semantic_activation_fingerprint != (
        second.semantic_activation_fingerprint
    )


def test_recovery_checkpoint_persists_semantic_activation_seal() -> None:
    authority = MutableSemanticAuthority({"topic_a": "source-a"})
    kernel, runtime = _activated_runtime(
        authority,
        CapturingExecutor(),
    )
    seal = kernel.seal_semantic_activation(runtime, ["topic_a"])
    session = SimpleNamespace(
        runtime=runtime,
        context_workspace=None,
        question_goal_contract=None,
        execution_graph_receipt=None,
        execution_graph_edges=[],
        query_branch_contexts={},
        artifact_goal_ids={},
        execution_graph_generation=0,
        execution_graph_fingerprint="",
        data_collection_sealed=False,
        analysis_skill_started=False,
        core_semantic_evidence=[],
    )

    payload = build_grounded_recovery_payload(
        session,
        thread_id="thread-1",
        run_id="run-1",
    )

    assert seal is not None
    assert payload["schemaVersion"] == 2
    assert payload["semanticActivation"]["seal"][
        "semanticActivationFingerprint"
    ] == seal.semantic_activation_fingerprint
    assert payload["semanticActivation"]["seal"][
        "exactTopics"
    ] == ["topic_a"]
