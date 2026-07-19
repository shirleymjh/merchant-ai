from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor

from merchant_ai.config import Settings
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_execution_graph import (
    GroundedExecutionEdgeSpec,
    GroundedExecutionGraphProposal,
    GroundedExecutionNodeSpec,
    build_grounded_execution_graph_receipt,
    grounded_execution_graph_fingerprint,
)
from merchant_ai.services.grounded_graph_revision_journal import (
    GroundedGraphRevisionJournalError,
    GroundedGraphRevisionTransactionJournal,
    build_grounded_graph_revision_recovery_payload,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationDynamicGraphEdge,
    PopulationDynamicGraphNode,
    PopulationDynamicGraphReceipt,
    seal_population_dynamic_graph_receipt,
)


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _workspace(tmp_path) -> GroundedContextWorkspace:
    settings = Settings(harness_workspace_path=str(tmp_path / "workspace"))
    return GroundedContextWorkspace.open(
        settings,
        thread_id="journal-thread",
        run_id="journal-run",
        merchant_id="journal-owner",
        access_role="journal-role",
        user_scope={},
        question="journal question",
    )


def _revision_inputs(tmp_path):
    workspace = _workspace(tmp_path)
    proposal = GroundedExecutionGraphProposal(
        base_version=1,
        goal_contract_fingerprint=_fingerprint("goal-contract"),
        discovery_snapshot_fingerprint=_fingerprint("discovery-snapshot"),
        nodes=[
            GroundedExecutionNodeSpec(
                client_key="source",
                objective="source objective",
                goal_ids=["goal-source"],
                topic_scope=["topic-a"],
                evidence_ref_ids=["semantic:topic-a:source"],
            ),
            GroundedExecutionNodeSpec(
                client_key="consumer",
                objective="consumer objective",
                goal_ids=["goal-consumer"],
                topic_scope=["topic-a"],
                evidence_ref_ids=["semantic:topic-a:consumer"],
            ),
        ],
        edges=[
            GroundedExecutionEdgeSpec(
                source_client_key="source",
                target_client_key="consumer",
                dependency_mode="VERIFIED_ARTIFACT",
                artifact_kind="VERIFIED_ENTITY_SET",
                target_binding_ref="semantic:topic-a:binding",
            )
        ],
    )
    execution_receipt = build_grounded_execution_graph_receipt(
        proposal,
        version=2,
    )
    population_receipt = seal_population_dynamic_graph_receipt(
        PopulationDynamicGraphReceipt(
            graph_id=execution_receipt.graph_id,
            graph_version=execution_receipt.version,
            graph_fingerprint=execution_receipt.fingerprint,
            nodes=tuple(
                PopulationDynamicGraphNode(
                    query_node_id=query_node_id,
                    consumer_goal_ids=tuple(
                        next(node.goal_ids for node in proposal.nodes if node.client_key == client_key)
                    ),
                )
                for client_key, query_node_id in (execution_receipt.node_ids.items())
            ),
            edges=(
                PopulationDynamicGraphEdge(
                    source_query_node_id=(execution_receipt.node_ids["source"]),
                    target_query_node_id=(execution_receipt.node_ids["consumer"]),
                    dependency_mode="VERIFIED_ARTIFACT",
                    artifact_kind="VERIFIED_ENTITY_SET",
                ),
            ),
            parent_receipt_fingerprint=_fingerprint("base-population-receipt"),
            revision_evidence_fingerprint=_fingerprint("evidence-set"),
        )
    )
    payload = build_grounded_graph_revision_recovery_payload(
        execution_proposal=proposal,
        execution_receipt=execution_receipt,
        population_receipt=population_receipt,
    )
    values = {
        "base_execution_receipt_fingerprint": _fingerprint("base-execution-receipt"),
        "new_execution_receipt_fingerprint": (grounded_execution_graph_fingerprint(proposal)),
        "base_population_receipt_fingerprint": _fingerprint("base-population-receipt"),
        "new_population_receipt_fingerprint": (population_receipt.receipt_fingerprint),
        "evidence_set_fingerprint": _fingerprint("evidence-set"),
        "recovery_payload": payload,
    }
    return workspace, values


def _error_code(call) -> str:
    try:
        call()
    except GroundedGraphRevisionJournalError as exc:
        return exc.code
    raise AssertionError("expected GroundedGraphRevisionJournalError")


def test_restart_load_returns_complete_prepared_recovery_payload(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    created = GroundedGraphRevisionTransactionJournal(workspace).prepare(**values)

    restarted = GroundedGraphRevisionTransactionJournal(workspace)
    recovery = restarted.load_recovery(created.record.transaction_id)

    assert created.committed is True
    assert recovery is not None
    assert recovery.phase == "PREPARED"
    assert recovery.next_action == "COMMIT_POPULATION"
    assert recovery.roll_forward_required is True
    assert recovery.recovery_payload == values["recovery_payload"]
    declarations = {item.client_key: item for item in recovery.recovery_payload.candidate_node_declarations}
    assert declarations["consumer"].initial_status == ("WAITING_VERIFIED_ENTITY_SET")
    assert declarations["consumer"].dependency_query_node_ids == (
        recovery.record.recovery_payload.execution_receipt["nodeIds"]["source"],
    )


def test_new_runtime_discovers_pending_without_retaining_transaction_id(
    tmp_path,
) -> None:
    first_workspace, values = _revision_inputs(tmp_path)
    created = GroundedGraphRevisionTransactionJournal(first_workspace).prepare(**values)

    restarted_workspace = _workspace(tmp_path)
    discovered = GroundedGraphRevisionTransactionJournal(restarted_workspace).discover_pending()

    assert len(discovered) == 1
    assert discovered[0].transaction_id == created.record.transaction_id
    assert discovered[0].phase == "PREPARED"
    assert discovered[0].next_action == "COMMIT_POPULATION"
    assert discovered[0].recovery_payload == values["recovery_payload"]


def test_pending_discovery_tracks_phase_and_excludes_completed_transaction(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    population = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(prepared.record.record_fingerprint),
    )

    restarted = GroundedGraphRevisionTransactionJournal(_workspace(tmp_path))
    pending = restarted.discover_pending()
    assert len(pending) == 1
    assert pending[0].phase == "POPULATION_COMMITTED"
    assert pending[0].next_action == "COMMIT_EXECUTION"

    journal.advance(
        prepared.record.transaction_id,
        target_status="EXECUTION_COMMITTED",
        expected_revision=population.record.revision,
        expected_record_fingerprint=(population.record.record_fingerprint),
    )
    assert restarted.discover_pending() == ()


def test_pending_discovery_materializes_claim_after_prepare_crash_window(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    transaction_path = journal.transaction_path(prepared.record.transaction_id)
    shutil.rmtree(transaction_path)

    restarted = GroundedGraphRevisionTransactionJournal(_workspace(tmp_path))
    pending = restarted.discover_pending()

    assert len(pending) == 1
    assert pending[0].transaction_id == prepared.record.transaction_id
    assert restarted.load(prepared.record.transaction_id) == (prepared.record)


def test_same_base_has_one_atomic_prepare_winner_and_no_loser_pending(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    competing_values = {
        **values,
        "evidence_set_fingerprint": _fingerprint("competing-evidence-set"),
    }

    def prepare(candidate):
        try:
            result = GroundedGraphRevisionTransactionJournal(workspace).prepare(**candidate)
            return ("COMMITTED", result.record.transaction_id)
        except GroundedGraphRevisionJournalError as exc:
            return ("REJECTED", exc.code)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                prepare,
                (values, competing_values),
            )
        )

    assert [item[0] for item in results].count("COMMITTED") == 1
    assert [item[0] for item in results].count("REJECTED") == 1
    assert next(item[1] for item in results if item[0] == "REJECTED") == "GRAPH_REVISION_BASE_ALREADY_CLAIMED"
    pending = GroundedGraphRevisionTransactionJournal(_workspace(tmp_path)).discover_pending()
    assert len(pending) == 1
    assert pending[0].transaction_id == next(item[1] for item in results if item[0] == "COMMITTED")


def test_journal_advances_in_order_and_exact_retries_are_idempotent(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    repeated_prepare = journal.prepare(**values)
    population = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(prepared.record.record_fingerprint),
    )
    repeated_population = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(prepared.record.record_fingerprint),
    )
    execution = journal.advance(
        prepared.record.transaction_id,
        target_status="EXECUTION_COMMITTED",
        expected_revision=population.record.revision,
        expected_record_fingerprint=(population.record.record_fingerprint),
    )

    assert repeated_prepare.idempotent is True
    assert population.committed is True
    assert repeated_population.idempotent is True
    assert repeated_population.record == population.record
    assert execution.record.status == "EXECUTION_COMMITTED"
    assert execution.record.parent_record_fingerprint == (population.record.record_fingerprint)
    recovery = journal.load_recovery(prepared.record.transaction_id)
    assert recovery is not None
    assert recovery.next_action == "COMPLETE"
    assert recovery.roll_forward_required is False


def test_status_jump_and_old_cas_are_rejected(tmp_path) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    jump_code = _error_code(
        lambda: journal.advance(
            prepared.record.transaction_id,
            target_status="EXECUTION_COMMITTED",
            expected_revision=prepared.record.revision,
            expected_record_fingerprint=(prepared.record.record_fingerprint),
        )
    )
    population = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(prepared.record.record_fingerprint),
    )
    stale_code = _error_code(
        lambda: journal.advance(
            prepared.record.transaction_id,
            target_status="EXECUTION_COMMITTED",
            expected_revision=prepared.record.revision,
            expected_record_fingerprint=(prepared.record.record_fingerprint),
        )
    )

    assert jump_code == "GRAPH_REVISION_STATUS_JUMP_FORBIDDEN"
    assert stale_code == "GRAPH_REVISION_JOURNAL_CAS_CONFLICT"
    assert journal.load(prepared.record.transaction_id) == (population.record)


def test_idempotent_advance_requires_the_exact_committed_predecessor(
    tmp_path,
) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    population = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(
            prepared.record.record_fingerprint
        ),
    )
    execution = journal.advance(
        prepared.record.transaction_id,
        target_status="EXECUTION_COMMITTED",
        expected_revision=population.record.revision,
        expected_record_fingerprint=(
            population.record.record_fingerprint
        ),
    )

    exact_older_retry = journal.advance(
        prepared.record.transaction_id,
        target_status="POPULATION_COMMITTED",
        expected_revision=prepared.record.revision,
        expected_record_fingerprint=(
            prepared.record.record_fingerprint
        ),
    )
    bogus_retry_code = _error_code(
        lambda: journal.advance(
            prepared.record.transaction_id,
            target_status="POPULATION_COMMITTED",
            expected_revision=999,
            expected_record_fingerprint="not-the-prepared-record",
        )
    )

    assert exact_older_retry.idempotent is True
    assert exact_older_retry.record == execution.record
    assert bogus_retry_code == "GRAPH_REVISION_JOURNAL_CAS_CONFLICT"


def test_concurrent_transition_has_one_commit_winner(tmp_path) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)

    def advance(_index: int):
        return GroundedGraphRevisionTransactionJournal(workspace).advance(
            prepared.record.transaction_id,
            target_status="POPULATION_COMMITTED",
            expected_revision=prepared.record.revision,
            expected_record_fingerprint=(prepared.record.record_fingerprint),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(advance, range(4)))

    assert sum(item.committed for item in results) == 1
    assert sum(item.idempotent for item in results) == 3
    assert len({item.record.record_fingerprint for item in results}) == 1


def test_tampered_head_and_unsealed_payload_fail_closed(tmp_path) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    unsealed = values["recovery_payload"].model_copy(
        update={"payload_fingerprint": ""},
        deep=True,
    )
    rejected = _error_code(
        lambda: journal.prepare(
            **{
                **values,
                "recovery_payload": unsealed,
            }
        )
    )
    assert rejected == "GRAPH_REVISION_RECOVERY_PAYLOAD_UNSEALED"

    prepared = journal.prepare(**values)
    head_path = journal.transaction_path(prepared.record.transaction_id) / "head.json"
    head = json.loads(head_path.read_text(encoding="utf-8"))
    head["status"] = "EXECUTION_COMMITTED"
    head_path.write_text(
        json.dumps(head, ensure_ascii=False),
        encoding="utf-8",
    )

    code = _error_code(lambda: GroundedGraphRevisionTransactionJournal(workspace).load(prepared.record.transaction_id))
    assert code == "GRAPH_REVISION_JOURNAL_HEAD_BINDING_INVALID"


def test_immutable_record_is_read_only_and_bound_to_head(tmp_path) -> None:
    workspace, values = _revision_inputs(tmp_path)
    journal = GroundedGraphRevisionTransactionJournal(workspace)
    prepared = journal.prepare(**values)
    transaction_root = journal.transaction_path(prepared.record.transaction_id)
    records = list(transaction_root.glob("record_*.json"))

    assert len(records) == 1
    assert records[0].stat().st_mode & 0o222 == 0
    assert journal.load(prepared.record.transaction_id) == prepared.record
