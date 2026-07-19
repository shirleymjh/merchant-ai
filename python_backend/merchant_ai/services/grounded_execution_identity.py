from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Mapping, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel, DataSnapshotContract, NodePlanContract
from merchant_ai.services.tool_runtime import ExecutionIdentity


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _required_text(value: Any, field_name: str) -> str:
    normalized = _text(value)
    if not normalized:
        raise ValueError("%s must not be empty" % field_name)
    return normalized


def _canonical_payload(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    return value


def _stable_fingerprint(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    for chunk in encoder.iterencode(_canonical_payload(value)):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


def _mapping_fingerprint(value: Mapping[str, Any] | None) -> str:
    return _stable_fingerprint(dict(value or {}))


def grounded_data_snapshot_identity(
    snapshot: DataSnapshotContract,
) -> dict[str, str]:
    """Return identity fields even when reusable snapshot caching is unsupported."""

    return {
        "datasourceFingerprint": _text(snapshot.datasource_fingerprint),
        "datasourceEnvironment": _text(snapshot.datasource_environment),
        "dataEpoch": _text(snapshot.data_epoch),
        "consistencyMode": _text(snapshot.consistency_mode),
        "semanticActivationFingerprint": _text(
            snapshot.semantic_activation_fingerprint
        ),
        "cacheGeneration": _text(snapshot.cache_generation),
    }


def grounded_data_snapshot_fingerprint(
    snapshot: DataSnapshotContract,
) -> str:
    return _stable_fingerprint(grounded_data_snapshot_identity(snapshot))


def grounded_access_contract_fingerprint(
    contract: NodePlanContract,
) -> str:
    """Bind the complete server-owned access contract, including row policy."""

    return _stable_fingerprint(contract)


class GroundedRunExecutionIdentitySeal(_StrictFrozenModel):
    """Content seal for the authority and activation shared by one run."""

    seal_version: Literal["grounded_run_execution_identity.v1"] = (
        "grounded_run_execution_identity.v1"
    )
    context_owner_fingerprint: str
    authorization_identity_fingerprint: str
    merchant_id: str
    tenant_id: str = ""
    principal_id: str = ""
    access_role: str = ""
    user_scope_fingerprint: str
    reference_scope_fingerprint: str
    datasource_fingerprint: str
    datasource_environment: str = ""
    semantic_activation_fingerprint: str
    cache_generation: str = ""
    seal_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedRunExecutionIdentitySeal":
        _required_text(
            self.context_owner_fingerprint,
            "context_owner_fingerprint",
        )
        _required_text(
            self.authorization_identity_fingerprint,
            "authorization_identity_fingerprint",
        )
        _required_text(self.merchant_id, "merchant_id")
        _required_text(self.user_scope_fingerprint, "user_scope_fingerprint")
        _required_text(
            self.reference_scope_fingerprint,
            "reference_scope_fingerprint",
        )
        _required_text(self.datasource_fingerprint, "datasource_fingerprint")
        _required_text(
            self.semantic_activation_fingerprint,
            "semantic_activation_fingerprint",
        )
        return self


def grounded_run_execution_identity_fingerprint(
    seal: GroundedRunExecutionIdentitySeal,
) -> str:
    payload = seal.model_dump(by_alias=True, mode="json")
    payload["sealFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_grounded_run_execution_identity(
    seal: GroundedRunExecutionIdentitySeal,
) -> GroundedRunExecutionIdentitySeal:
    return seal.model_copy(
        update={
            "seal_fingerprint": grounded_run_execution_identity_fingerprint(
                seal
            )
        }
    )


def grounded_run_execution_identity_valid(
    seal: GroundedRunExecutionIdentitySeal,
) -> bool:
    declared = _text(seal.seal_fingerprint)
    return bool(
        declared
        and declared == grounded_run_execution_identity_fingerprint(seal)
    )


def build_grounded_run_execution_identity(
    *,
    context_owner_fingerprint: str,
    execution_identity: ExecutionIdentity,
    user_scope: Mapping[str, Any] | None,
    reference_scope: Mapping[str, Any] | APIModel | None,
    datasource_fingerprint: str,
    cache_generation: str = "",
) -> GroundedRunExecutionIdentitySeal:
    if isinstance(reference_scope, APIModel):
        reference_payload = reference_scope.model_dump(
            by_alias=True,
            mode="json",
        )
    elif isinstance(reference_scope, Mapping):
        reference_payload = dict(reference_scope)
    else:
        reference_payload = {}
    seal = GroundedRunExecutionIdentitySeal(
        context_owner_fingerprint=_text(context_owner_fingerprint),
        authorization_identity_fingerprint=(
            execution_identity.fingerprint().key()
        ),
        merchant_id=_text(execution_identity.merchant_id),
        tenant_id=_text(execution_identity.tenant_id),
        principal_id=_text(execution_identity.principal_id),
        access_role=_text(execution_identity.role),
        user_scope_fingerprint=_mapping_fingerprint(user_scope),
        reference_scope_fingerprint=_mapping_fingerprint(reference_payload),
        datasource_fingerprint=_text(datasource_fingerprint),
        datasource_environment=_text(
            execution_identity.datasource_environment
        ),
        semantic_activation_fingerprint=_text(
            execution_identity.semantic_activation_fingerprint
        ),
        cache_generation=_text(cache_generation),
    )
    return seal_grounded_run_execution_identity(seal)


class GroundedNodeExecutionIdentitySeal(_StrictFrozenModel):
    """Replay-resistant identity for one dynamically chosen query node."""

    seal_version: Literal["grounded_node_execution_identity.v1"] = (
        "grounded_node_execution_identity.v1"
    )
    run_identity_fingerprint: str
    graph_fingerprint: str
    query_node_id: str
    generation: int = Field(ge=1)
    attempt_id: str
    goal_contract_fingerprint: str
    query_contract_fingerprint: str
    sql_ast_fingerprint: str
    data_snapshot_fingerprint: str
    access_contract_fingerprints: tuple[str, ...]
    seal_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "GroundedNodeExecutionIdentitySeal":
        for field_name in (
            "run_identity_fingerprint",
            "graph_fingerprint",
            "query_node_id",
            "attempt_id",
            "goal_contract_fingerprint",
            "query_contract_fingerprint",
            "sql_ast_fingerprint",
            "data_snapshot_fingerprint",
        ):
            _required_text(getattr(self, field_name), field_name)
        if not self.access_contract_fingerprints:
            raise ValueError("access_contract_fingerprints must not be empty")
        normalized = tuple(
            _required_text(value, "access_contract_fingerprints")
            for value in self.access_contract_fingerprints
        )
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "access_contract_fingerprints must not contain duplicates"
            )
        return self


def grounded_node_execution_identity_fingerprint(
    seal: GroundedNodeExecutionIdentitySeal,
) -> str:
    payload = seal.model_dump(by_alias=True, mode="json")
    payload["sealFingerprint"] = ""
    return _stable_fingerprint(payload)


def seal_grounded_node_execution_identity(
    seal: GroundedNodeExecutionIdentitySeal,
) -> GroundedNodeExecutionIdentitySeal:
    return seal.model_copy(
        update={
            "seal_fingerprint": grounded_node_execution_identity_fingerprint(
                seal
            )
        }
    )


def grounded_node_execution_identity_valid(
    seal: GroundedNodeExecutionIdentitySeal,
    *,
    run_identity: GroundedRunExecutionIdentitySeal,
) -> bool:
    declared = _text(seal.seal_fingerprint)
    return bool(
        grounded_run_execution_identity_valid(run_identity)
        and seal.run_identity_fingerprint == run_identity.seal_fingerprint
        and declared
        and declared == grounded_node_execution_identity_fingerprint(seal)
    )


def build_grounded_node_execution_identity(
    *,
    run_identity: GroundedRunExecutionIdentitySeal,
    graph_fingerprint: str,
    query_node_id: str,
    generation: int,
    attempt_id: str,
    goal_contract_fingerprint: str,
    query_contract_fingerprint: str,
    sql_ast_fingerprint: str,
    data_snapshot: DataSnapshotContract,
    access_contracts: Sequence[NodePlanContract],
) -> GroundedNodeExecutionIdentitySeal:
    if not grounded_run_execution_identity_valid(run_identity):
        raise ValueError("run_identity must have a valid server content seal")
    snapshot_identity = grounded_data_snapshot_identity(data_snapshot)
    expected_snapshot_identity = {
        "datasourceFingerprint": run_identity.datasource_fingerprint,
        "datasourceEnvironment": run_identity.datasource_environment,
        "semanticActivationFingerprint": (
            run_identity.semantic_activation_fingerprint
        ),
        "cacheGeneration": run_identity.cache_generation,
    }
    for field_name, expected in expected_snapshot_identity.items():
        if snapshot_identity[field_name] != expected:
            raise ValueError(
                "data snapshot does not match run identity: %s" % field_name
            )
    access_fingerprints = tuple(
        sorted(
            {
                grounded_access_contract_fingerprint(contract)
                for contract in access_contracts
            }
        )
    )
    seal = GroundedNodeExecutionIdentitySeal(
        run_identity_fingerprint=run_identity.seal_fingerprint,
        graph_fingerprint=_text(graph_fingerprint),
        query_node_id=_text(query_node_id),
        generation=int(generation),
        attempt_id=_text(attempt_id),
        goal_contract_fingerprint=_text(goal_contract_fingerprint),
        query_contract_fingerprint=_text(query_contract_fingerprint),
        sql_ast_fingerprint=_text(sql_ast_fingerprint),
        data_snapshot_fingerprint=grounded_data_snapshot_fingerprint(
            data_snapshot
        ),
        access_contract_fingerprints=access_fingerprints,
    )
    return seal_grounded_node_execution_identity(seal)


GroundedExecutionIdentityGateStage = Literal[
    "PRE_EXECUTION",
    "PRE_PUBLICATION",
]


class GroundedExecutionIdentityGateResult(_StrictFrozenModel):
    """Server-only result of replaying a seal against current authority.

    Validating the stored content hashes alone only proves that the stored
    objects were not mutated.  It does not prove that the active merchant,
    access scope, semantic activation, graph, SQL or data snapshot still
    matches them.  This result is produced by rebuilding both seals from live
    server inputs at each irreversible gate.
    """

    stage: GroundedExecutionIdentityGateStage
    passed: bool
    code: str
    stored_run_identity_fingerprint: str = ""
    live_run_identity_fingerprint: str = ""
    stored_node_identity_fingerprint: str = ""
    live_node_identity_fingerprint: str = ""


class GroundedExecutionIdentityGateError(RuntimeError):
    def __init__(self, result: GroundedExecutionIdentityGateResult) -> None:
        self.result = result
        super().__init__(result.code)


def verify_grounded_execution_identity_live(
    *,
    stage: GroundedExecutionIdentityGateStage,
    stored_run_identity: GroundedRunExecutionIdentitySeal,
    stored_node_identity: GroundedNodeExecutionIdentitySeal,
    context_owner_fingerprint: str,
    execution_identity: ExecutionIdentity,
    user_scope: Mapping[str, Any] | None,
    reference_scope: Mapping[str, Any] | APIModel | None,
    datasource_fingerprint: str,
    cache_generation: str,
    graph_fingerprint: str,
    query_node_id: str,
    generation: int,
    attempt_id: str,
    goal_contract_fingerprint: str,
    query_contract_fingerprint: str,
    sql_ast_fingerprint: str,
    data_snapshot: DataSnapshotContract,
    access_contracts: Sequence[NodePlanContract],
) -> GroundedExecutionIdentityGateResult:
    """Rebuild run and node identity from live values and fail closed.

    Callers must invoke this once immediately before Doris execution and once
    immediately before making a verified artifact visible.  No value from the
    model, an old branch object, or a staged artifact is accepted as a live
    authority input.
    """

    stored_run_fingerprint = _text(stored_run_identity.seal_fingerprint)
    stored_node_fingerprint = _text(stored_node_identity.seal_fingerprint)
    base = {
        "stage": stage,
        "stored_run_identity_fingerprint": stored_run_fingerprint,
        "stored_node_identity_fingerprint": stored_node_fingerprint,
    }
    if not grounded_run_execution_identity_valid(stored_run_identity):
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_STORED_RUN_INVALID",
        )
    if not grounded_node_execution_identity_valid(
        stored_node_identity,
        run_identity=stored_run_identity,
    ):
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_STORED_NODE_INVALID",
        )
    try:
        live_run_identity = build_grounded_run_execution_identity(
            context_owner_fingerprint=context_owner_fingerprint,
            execution_identity=execution_identity,
            user_scope=user_scope,
            reference_scope=reference_scope,
            datasource_fingerprint=datasource_fingerprint,
            cache_generation=cache_generation,
        )
    except Exception:
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_LIVE_RUN_INVALID",
        )
    live_run_fingerprint = live_run_identity.seal_fingerprint
    if live_run_fingerprint != stored_run_fingerprint:
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_RUN_DRIFT",
            live_run_identity_fingerprint=live_run_fingerprint,
        )
    try:
        live_node_identity = build_grounded_node_execution_identity(
            run_identity=live_run_identity,
            graph_fingerprint=graph_fingerprint,
            query_node_id=query_node_id,
            generation=generation,
            attempt_id=attempt_id,
            goal_contract_fingerprint=goal_contract_fingerprint,
            query_contract_fingerprint=query_contract_fingerprint,
            sql_ast_fingerprint=sql_ast_fingerprint,
            data_snapshot=data_snapshot,
            access_contracts=access_contracts,
        )
    except Exception:
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_LIVE_NODE_INVALID",
            live_run_identity_fingerprint=live_run_fingerprint,
        )
    live_node_fingerprint = live_node_identity.seal_fingerprint
    if live_node_fingerprint != stored_node_fingerprint:
        return GroundedExecutionIdentityGateResult(
            **base,
            passed=False,
            code="EXECUTION_IDENTITY_NODE_DRIFT",
            live_run_identity_fingerprint=live_run_fingerprint,
            live_node_identity_fingerprint=live_node_fingerprint,
        )
    return GroundedExecutionIdentityGateResult(
        **base,
        passed=True,
        code="EXECUTION_IDENTITY_MATCHED",
        live_run_identity_fingerprint=live_run_fingerprint,
        live_node_identity_fingerprint=live_node_fingerprint,
    )


def require_grounded_execution_identity_live(
    **kwargs: Any,
) -> GroundedExecutionIdentityGateResult:
    result = verify_grounded_execution_identity_live(**kwargs)
    if not result.passed:
        raise GroundedExecutionIdentityGateError(result)
    return result
