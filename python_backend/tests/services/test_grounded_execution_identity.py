from __future__ import annotations

import pytest

from merchant_ai.models import DataSnapshotContract, NodePlanContract
from merchant_ai.services.grounded_execution_identity import (
    GroundedExecutionIdentityGateError,
    build_grounded_node_execution_identity,
    build_grounded_run_execution_identity,
    grounded_access_contract_fingerprint,
    grounded_data_snapshot_fingerprint,
    grounded_node_execution_identity_valid,
    grounded_run_execution_identity_valid,
    require_grounded_execution_identity_live,
    seal_grounded_run_execution_identity,
    verify_grounded_execution_identity_live,
)
from merchant_ai.services.tool_runtime import ExecutionIdentity


def _execution_identity(
    *,
    merchant_id: str = "merchant-a",
    semantic_activation_fingerprint: str = "activation-a",
) -> ExecutionIdentity:
    return ExecutionIdentity.from_server_context(
        merchant_id=merchant_id,
        tenant_id="tenant-a",
        principal_id="principal-a",
        role="analyst",
        permissions=("query", "artifact-read"),
        region="region-a",
        store_ids=("scope-a", "scope-b"),
        row_policy={"policyVersion": "policy-a"},
        datasource_environment="production-a",
        semantic_activation_fingerprint=semantic_activation_fingerprint,
    )


def _run_identity(
    *,
    merchant_id: str = "merchant-a",
    semantic_activation_fingerprint: str = "activation-a",
):
    return build_grounded_run_execution_identity(
        context_owner_fingerprint="context-owner-a",
        execution_identity=_execution_identity(
            merchant_id=merchant_id,
            semantic_activation_fingerprint=(
                semantic_activation_fingerprint
            ),
        ),
        user_scope={"authorizedScope": ["scope-a", "scope-b"]},
        reference_scope={"topicRefs": ["topic-a"]},
        datasource_fingerprint="datasource-a",
        cache_generation="cache-a",
    )


def _snapshot(
    *,
    datasource_fingerprint: str = "datasource-a",
    semantic_activation_fingerprint: str = "activation-a",
) -> DataSnapshotContract:
    return DataSnapshotContract(
        datasource_fingerprint=datasource_fingerprint,
        datasource_environment="production-a",
        consistency_mode="UNSUPPORTED",
        semantic_activation_fingerprint=semantic_activation_fingerprint,
        cache_generation="cache-a",
        unsupported_reason="EPOCH_CAPABILITY_UNAVAILABLE",
    )


def _access_contract(
    *,
    table: str = "table-a",
    policy_version: str = "row-policy-a",
) -> NodePlanContract:
    return NodePlanContract(
        task_id="query-node-a",
        preferred_table=table,
        allowed_columns=["entity-id", "value"],
        row_scope_policy={"policyVersion": policy_version},
    )


def _node_identity(*, run_identity=None, snapshot=None, access_contracts=None):
    return build_grounded_node_execution_identity(
        run_identity=run_identity or _run_identity(),
        graph_fingerprint="graph-a",
        query_node_id="query-node-a",
        generation=4,
        attempt_id="attempt-a",
        goal_contract_fingerprint="goal-contract-a",
        query_contract_fingerprint="query-contract-a",
        sql_ast_fingerprint="sql-ast-a",
        data_snapshot=snapshot or _snapshot(),
        access_contracts=(
            (_access_contract(),)
            if access_contracts is None
            else access_contracts
        ),
    )


def _live_gate_kwargs(*, stage: str = "PRE_EXECUTION") -> dict[str, object]:
    run_identity = _run_identity()
    node_identity = _node_identity(run_identity=run_identity)
    return {
        "stage": stage,
        "stored_run_identity": run_identity,
        "stored_node_identity": node_identity,
        "context_owner_fingerprint": "context-owner-a",
        "execution_identity": _execution_identity(),
        "user_scope": {"authorizedScope": ["scope-a", "scope-b"]},
        "reference_scope": {"topicRefs": ["topic-a"]},
        "datasource_fingerprint": "datasource-a",
        "cache_generation": "cache-a",
        "graph_fingerprint": "graph-a",
        "query_node_id": "query-node-a",
        "generation": 4,
        "attempt_id": "attempt-a",
        "goal_contract_fingerprint": "goal-contract-a",
        "query_contract_fingerprint": "query-contract-a",
        "sql_ast_fingerprint": "sql-ast-a",
        "data_snapshot": _snapshot(),
        "access_contracts": (_access_contract(),),
    }


def test_run_and_node_identity_are_valid_when_every_binding_matches() -> None:
    run_identity = _run_identity()
    node_identity = _node_identity(run_identity=run_identity)

    assert grounded_run_execution_identity_valid(run_identity)
    assert grounded_node_execution_identity_valid(
        node_identity,
        run_identity=run_identity,
    )


def test_unsupported_snapshot_still_binds_datasource_and_activation() -> None:
    first = _snapshot()
    changed_datasource = _snapshot(datasource_fingerprint="datasource-b")
    changed_activation = _snapshot(
        semantic_activation_fingerprint="activation-b"
    )

    assert first.cache_identity() == {}
    assert grounded_data_snapshot_fingerprint(first)
    assert grounded_data_snapshot_fingerprint(first) != (
        grounded_data_snapshot_fingerprint(changed_datasource)
    )
    assert grounded_data_snapshot_fingerprint(first) != (
        grounded_data_snapshot_fingerprint(changed_activation)
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("merchant_id", "merchant-b"),
        ("semantic_activation_fingerprint", "activation-b"),
        ("datasource_fingerprint", "datasource-b"),
        ("cache_generation", "cache-b"),
        ("context_owner_fingerprint", "context-owner-b"),
    ),
)
def test_run_identity_change_invalidates_existing_node(
    field_name: str,
    changed_value: str,
) -> None:
    run_identity = _run_identity()
    node_identity = _node_identity(run_identity=run_identity)
    changed_run = seal_grounded_run_execution_identity(
        run_identity.model_copy(update={field_name: changed_value})
    )

    assert grounded_run_execution_identity_valid(changed_run)
    assert not grounded_node_execution_identity_valid(
        node_identity,
        run_identity=changed_run,
    )


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("graph_fingerprint", "graph-b"),
        ("query_node_id", "query-node-b"),
        ("generation", 5),
        ("attempt_id", "attempt-b"),
        ("goal_contract_fingerprint", "goal-contract-b"),
        ("query_contract_fingerprint", "query-contract-b"),
        ("sql_ast_fingerprint", "sql-ast-b"),
        ("data_snapshot_fingerprint", "snapshot-b"),
    ),
)
def test_node_identity_mutation_without_a_new_server_seal_is_rejected(
    field_name: str,
    changed_value: object,
) -> None:
    run_identity = _run_identity()
    node_identity = _node_identity(run_identity=run_identity)
    mutated = node_identity.model_copy(update={field_name: changed_value})

    assert not grounded_node_execution_identity_valid(
        mutated,
        run_identity=run_identity,
    )


def test_snapshot_cannot_cross_run_datasource_or_activation() -> None:
    run_identity = _run_identity()

    with pytest.raises(ValueError):
        _node_identity(
            run_identity=run_identity,
            snapshot=_snapshot(datasource_fingerprint="datasource-b"),
        )
    with pytest.raises(ValueError):
        _node_identity(
            run_identity=run_identity,
            snapshot=_snapshot(
                semantic_activation_fingerprint="activation-b"
            ),
        )


def test_access_contract_seal_covers_table_and_row_policy() -> None:
    first = _access_contract()
    changed_table = _access_contract(table="table-b")
    changed_policy = _access_contract(policy_version="row-policy-b")

    assert grounded_access_contract_fingerprint(first) != (
        grounded_access_contract_fingerprint(changed_table)
    )
    assert grounded_access_contract_fingerprint(first) != (
        grounded_access_contract_fingerprint(changed_policy)
    )


def test_node_requires_at_least_one_server_access_contract() -> None:
    with pytest.raises(ValueError):
        _node_identity(access_contracts=())


@pytest.mark.parametrize("stage", ("PRE_EXECUTION", "PRE_PUBLICATION"))
def test_live_gate_replays_every_identity_at_both_irreversible_gates(
    stage: str,
) -> None:
    result = verify_grounded_execution_identity_live(
        **_live_gate_kwargs(stage=stage)
    )

    assert result.passed is True
    assert result.code == "EXECUTION_IDENTITY_MATCHED"
    assert result.live_run_identity_fingerprint == (
        result.stored_run_identity_fingerprint
    )
    assert result.live_node_identity_fingerprint == (
        result.stored_node_identity_fingerprint
    )


def test_live_gate_rejects_a_tampered_stored_run_seal() -> None:
    values = _live_gate_kwargs()
    values["stored_run_identity"] = values[
        "stored_run_identity"
    ].model_copy(update={"merchant_id": "merchant-b"})

    result = verify_grounded_execution_identity_live(**values)

    assert result.passed is False
    assert result.code == "EXECUTION_IDENTITY_STORED_RUN_INVALID"


def test_live_gate_rejects_a_tampered_stored_node_seal() -> None:
    values = _live_gate_kwargs()
    values["stored_node_identity"] = values[
        "stored_node_identity"
    ].model_copy(update={"sql_ast_fingerprint": "sql-ast-b"})

    result = verify_grounded_execution_identity_live(**values)

    assert result.passed is False
    assert result.code == "EXECUTION_IDENTITY_STORED_NODE_INVALID"


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("context_owner_fingerprint", "context-owner-b"),
        ("execution_identity", _execution_identity(merchant_id="merchant-b")),
        ("user_scope", {"authorizedScope": ["scope-a"]}),
        ("reference_scope", {"topicRefs": ["topic-b"]}),
        ("datasource_fingerprint", "datasource-b"),
        ("cache_generation", "cache-b"),
    ),
)
def test_live_gate_rejects_current_run_authority_drift(
    field_name: str,
    changed_value: object,
) -> None:
    values = _live_gate_kwargs()
    values[field_name] = changed_value

    result = verify_grounded_execution_identity_live(**values)

    assert result.passed is False
    assert result.code == "EXECUTION_IDENTITY_RUN_DRIFT"


@pytest.mark.parametrize(
    ("field_name", "changed_value"),
    (
        ("graph_fingerprint", "graph-b"),
        ("query_node_id", "query-node-b"),
        ("generation", 5),
        ("attempt_id", "attempt-b"),
        ("goal_contract_fingerprint", "goal-contract-b"),
        ("query_contract_fingerprint", "query-contract-b"),
        ("sql_ast_fingerprint", "sql-ast-b"),
        ("access_contracts", (_access_contract(table="table-b"),)),
    ),
)
def test_live_gate_rejects_current_node_binding_drift(
    field_name: str,
    changed_value: object,
) -> None:
    values = _live_gate_kwargs()
    values[field_name] = changed_value

    result = verify_grounded_execution_identity_live(**values)

    assert result.passed is False
    assert result.code == "EXECUTION_IDENTITY_NODE_DRIFT"


def test_live_gate_rejects_snapshot_that_no_longer_matches_run() -> None:
    values = _live_gate_kwargs()
    values["data_snapshot"] = _snapshot(
        semantic_activation_fingerprint="activation-b"
    )

    result = verify_grounded_execution_identity_live(**values)

    assert result.passed is False
    assert result.code == "EXECUTION_IDENTITY_LIVE_NODE_INVALID"


def test_required_live_gate_raises_typed_failure_without_exposing_payloads() -> None:
    values = _live_gate_kwargs(stage="PRE_PUBLICATION")
    values["sql_ast_fingerprint"] = "sql-ast-b"

    with pytest.raises(GroundedExecutionIdentityGateError) as captured:
        require_grounded_execution_identity_live(**values)

    assert captured.value.result.code == "EXECUTION_IDENTITY_NODE_DRIFT"
    assert str(captured.value) == "EXECUTION_IDENTITY_NODE_DRIFT"
