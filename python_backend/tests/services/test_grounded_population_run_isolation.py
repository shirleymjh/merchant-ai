from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merchant_ai.config import Settings
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_population_runtime_gate import (
    GroundedPopulationExecutionGate,
    GroundedPopulationRuntimeGateError,
)


def _fingerprint(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _gate(settings: Settings) -> GroundedPopulationExecutionGate:
    return GroundedPopulationExecutionGate(
        settings=settings,
        semantic_provider=object(),
        declaration_author_fingerprint=_fingerprint("declaration"),
        semantic_authority_fingerprint=_fingerprint("semantic"),
        lineage_authority_fingerprint=_fingerprint("lineage"),
        artifact_authority_fingerprint=_fingerprint("artifact"),
        ledger_authority_fingerprint=_fingerprint("ledger"),
        semantic_timeout_seconds=1.0,
    )


def _workspace(
    settings: Settings,
    *,
    run_id: str,
) -> GroundedContextWorkspace:
    return GroundedContextWorkspace.open(
        settings,
        thread_id="shared-thread",
        run_id=run_id,
        merchant_id="shared-principal",
        access_role="shared-role",
        user_scope={"region": "shared-region"},
        question="same question",
    )


def test_same_owner_and_question_use_distinct_run_gate_authorities(
    tmp_path: Path,
) -> None:
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace")
    )
    gate = _gate(settings)
    first = _workspace(settings, run_id="run-first")
    second = _workspace(settings, run_id="run-second")

    gate.register_run(workspace=first, ledger_provider=lambda: ())
    gate.register_run(workspace=second, ledger_provider=lambda: ())

    assert first.owner_fingerprint == second.owner_fingerprint
    assert first.request_fingerprint != second.request_fingerprint
    first_gate_id = gate.gate_id(
        context_owner_fingerprint=first.owner_fingerprint,
        run_authority_fingerprint=first.request_fingerprint,
        goal_contract_fingerprint=_fingerprint("same-goal"),
    )
    second_gate_id = gate.gate_id(
        context_owner_fingerprint=second.owner_fingerprint,
        run_authority_fingerprint=second.request_fingerprint,
        goal_contract_fingerprint=_fingerprint("same-goal"),
    )

    assert first_gate_id != second_gate_id
    assert gate._run_authority(
        first.owner_fingerprint,
        first.request_fingerprint,
    ).workspace.root == first.root
    assert gate._run_authority(
        second.owner_fingerprint,
        second.request_fingerprint,
    ).workspace.root == second.root


def test_concurrent_same_owner_runs_do_not_collide_or_cross_resolve(
    tmp_path: Path,
) -> None:
    settings = Settings(
        harness_workspace_path=str(tmp_path / "workspace")
    )
    gate = _gate(settings)
    workspaces = tuple(
        _workspace(settings, run_id="run-%d" % index)
        for index in range(8)
    )

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = tuple(
            pool.submit(
                gate.register_run,
                workspace=workspace,
                ledger_provider=lambda: (),
            )
            for workspace in workspaces
        )
        for future in futures:
            future.result()

    resolved_roots = {
        gate._run_authority(
            workspace.owner_fingerprint,
            workspace.request_fingerprint,
        ).workspace.root
        for workspace in workspaces
    }
    assert resolved_roots == {workspace.root for workspace in workspaces}
    with pytest.raises(GroundedPopulationRuntimeGateError) as raised:
        gate._run_authority(
            _fingerprint("different-owner"),
            workspaces[0].request_fingerprint,
        )
    assert str(raised.value) == "POPULATION_RUN_AUTHORITY_NOT_REGISTERED"
