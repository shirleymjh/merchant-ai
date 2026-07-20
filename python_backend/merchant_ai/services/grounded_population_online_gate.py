from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Optional, Sequence

from pydantic import ConfigDict, Field, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.artifacts import WorkspaceArtifactStore
from merchant_ai.services.grounded_context_workspace import (
    GroundedContextWorkspace,
)
from merchant_ai.services.grounded_execution_identity import (
    grounded_data_snapshot_fingerprint,
)
from merchant_ai.services.grounded_goal_contract import (
    OriginalQuestionGoalContract,
    parse_original_question_goal_contract,
)
from merchant_ai.services.grounded_population_gate_coordinator import (
    PopulationArtifactLedgerEntry,
    PopulationArtifactLedgerSnapshot,
    PopulationExecutionGraphBinding,
    PopulationGateCoordinator,
    PopulationGatePhase,
    PopulationGateState,
    PopulationGateStateStore,
    PopulationGateTransitionResult,
    PopulationGoalDeclarationCommand,
    PopulationNodePostResultCommand,
    PopulationNodePreExecutionCommand,
    PopulationPostResultCommand,
    PopulationPreExecutionCommand,
    PopulationPublishedArtifactReceipt,
    PopulationResultSelection,
    population_dynamic_graph_receipt_fingerprint,
    population_execution_graph_binding_fingerprint,
    population_gate_state_fingerprint,
    population_node_gate_record_fingerprint,
    seal_population_artifact_ledger_entry,
    seal_population_artifact_ledger_snapshot,
    seal_population_published_artifact_receipt,
)
from merchant_ai.services.grounded_population_semantic_reviewer import (
    IndependentPopulationSemanticReviewer,
    PopulationSemanticProviderDecision,
    PopulationSemanticProviderOutput,
    PopulationSemanticReviewerOutcome,
    PopulationSemanticReviewerRequest,
    goal_declaration_population_input_from_review,
    population_goal_skeleton_fingerprint,
    population_semantic_reviewer_request_fingerprint,
)
from merchant_ai.services.grounded_population_verifier import (
    PopulationArtifactCoverage,
    PopulationArtifactEvidence,
    PopulationArtifactKind,
    PopulationExecutionClaim,
    PopulationScopeKind,
    PopulationVerificationAttestation,
    PopulationVerificationStage,
    population_attestation_fingerprint,
)

if TYPE_CHECKING:
    from merchant_ai.services.grounded_runtime_kernel import (
        GroundedVerifiedQueryArtifact,
    )


class PopulationOnlineGateStorageError(RuntimeError):
    def __init__(self, code: str):
        self.code = str(code or "POPULATION_GATE_STORAGE_FAILED")
        super().__init__(self.code)


class PopulationOnlineLedgerError(RuntimeError):
    def __init__(self, code: str):
        self.code = str(code or "POPULATION_LEDGER_READ_FAILED")
        super().__init__(self.code)


class _StrictFrozenModel(APIModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _enum_value(value: Any) -> str:
    return _text(getattr(value, "value", value))


def _stable_json_bytes(value: Any) -> bytes:
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True, mode="json")
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _stable_fingerprint(value: Any) -> str:
    return hashlib.sha256(_stable_json_bytes(value)).hexdigest()


def _valid_sha256(value: Any) -> bool:
    candidate = str(value or "")
    return len(candidate) == 64 and all(character in "0123456789abcdef" for character in candidate)


def _safe_relative_path(value: Any) -> str:
    raw = _text(value).replace("\\", "/")
    path = Path(raw)
    if not raw or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise PopulationOnlineLedgerError("POPULATION_LEDGER_ARTIFACT_PATH_INVALID")
    return path.as_posix()


class PopulationZeroToolModelDecision(_StrictFrozenModel):
    """Only semantic decisions come from the model; bindings stay server-owned."""

    complete: bool
    decisions: tuple[PopulationSemanticProviderDecision, ...]


class StructuredPopulationSemanticModelProvider:
    """Strict structured model provider with no tool, SQL, or asset surface."""

    _SYSTEM_PROMPT = """You are an isolated population-semantics reviewer.
You receive the exact original question and a population-blind Goal skeleton.
For every Goal, decide whether a population gate is structurally required.
A population gate is required only when that Goal must consume or preserve a row/entity population defined by another Goal, a verified predicate scope, or a verified result artifact.
Do not create a population gate merely because a Goal is a metric, has a time window, is grouped by a dimension, participates in a comparison, or is referenced by answer-composition fields. In particular, TIME_WINDOW.appliesToGoalIds, RANKING.metricGoalIds/dimensionGoalIds, COMPARISON.leftGoalIds/rightGoalIds, and ANALYSIS.inputGoalIds/baselineGoalIds do not by themselves mean population dependence.
For a standalone metric such as "最近7天订单总数", both the METRIC Goal and TIME_WINDOW Goal must return gateRequired=false with no scopeKind and no sourceGoalIds.
For a dependent request such as "先找销量最高的3个商品，再查这些商品的退款量", the downstream Goal is population-gated and must reference the upstream entity-producing Goal.
UNIVERSE is not a distinct declaration in this protocol. A standalone query over all matching rows is not gated. If a direct non-dependent scope truly must be tracked, use INDEPENDENT rather than UNIVERSE.
When a gate is required, return one typed scope kind and only the source Goal IDs required by that kind.
Use only supplied Goal IDs. Do not infer tables, fields, formulas, SQL, assets, execution nodes, query topology, row values, or business rules. Return only the strict structured decision schema."""

    def __init__(self, model: Any, *, authority_fingerprint: str) -> None:
        authority = _text(authority_fingerprint)
        if model is None:
            raise ValueError("structured population model is required")
        if not authority:
            raise ValueError("authority_fingerprint is required")
        self.model = model
        self._authority_fingerprint = authority

    @property
    def authority_fingerprint(self) -> str:
        return self._authority_fingerprint

    def review_population_semantics(
        self,
        request: PopulationSemanticReviewerRequest,
        *,
        timeout_seconds: float,
    ) -> PopulationSemanticProviderOutput:
        if float(timeout_seconds) <= 0:
            raise ValueError("timeout_seconds must be positive")
        if request.request_fingerprint != population_semantic_reviewer_request_fingerprint(
            request
        ) or request.goal_skeleton_fingerprint != population_goal_skeleton_fingerprint(request.goal_skeleton):
            raise RuntimeError("POPULATION_MODEL_REQUEST_BINDING_INVALID")
        bind = getattr(self.model, "with_structured_output", None)
        if not callable(bind):
            raise RuntimeError("STRUCTURED_POPULATION_MODEL_REQUIRED")
        structured_model = bind(
            PopulationZeroToolModelDecision,
            method="json_schema",
            strict=True,
        )
        model_payload = {
            "question": request.effective_question,
            "goalSkeleton": [item.model_dump(by_alias=True, mode="json") for item in request.goal_skeleton],
        }
        raw = structured_model.invoke(
            [
                ("system", self._SYSTEM_PROMPT),
                (
                    "human",
                    _stable_json_bytes(model_payload).decode("utf-8"),
                ),
            ]
        )
        decision = self._parse_decision(raw)
        return PopulationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            goal_skeleton_fingerprint=(request.goal_skeleton_fingerprint),
            complete=decision.complete,
            decisions=self._canonical_decisions(decision.decisions),
        )

    @staticmethod
    def _canonical_decisions(
        decisions: Sequence[PopulationSemanticProviderDecision],
    ) -> tuple[PopulationSemanticProviderDecision, ...]:
        """Translate model vocabulary into the Goal declaration vocabulary.

        Goal Contracts represent a direct all-matching-rows scope as
        ``ALL_MATCHING_ROWS``, which the population verifier projects to
        ``INDEPENDENT``.  The model schema historically also exposed
        ``UNIVERSE`` even though Core cannot declare that value.  Both scopes
        use direct-scope lineage at execution time, so retaining ``UNIVERSE``
        would create a protocol-only mismatch and reject an otherwise safe
        query.  Canonicalize that alias before sealing the provider output.
        """

        canonical: list[PopulationSemanticProviderDecision] = []
        for item in decisions:
            if (
                item.gate_required
                and item.scope_kind == PopulationScopeKind.UNIVERSE
            ):
                item = item.model_copy(
                    update={"scope_kind": PopulationScopeKind.INDEPENDENT}
                )
            canonical.append(item)
        return tuple(canonical)

    @staticmethod
    def _parse_decision(value: Any) -> PopulationZeroToolModelDecision:
        if isinstance(value, PopulationZeroToolModelDecision):
            return value
        if isinstance(value, Mapping):
            return PopulationZeroToolModelDecision.model_validate(dict(value))
        content: Optional[Any] = getattr(value, "content", None)
        if isinstance(content, Mapping):
            return PopulationZeroToolModelDecision.model_validate(dict(content))
        raise TypeError("structured population model returned an invalid value")


class _PopulationCheckpointHead(_StrictFrozenModel):
    head_version: str = "population_gate_checkpoint_head.v1"
    owner_fingerprint: str
    checkpoint_namespace_fingerprint: str
    gate_id_fingerprint: str
    revision: int = Field(ge=1)
    state_fingerprint: str
    state_artifact_sha256: str
    head_fingerprint: str = ""

    @model_validator(mode="after")
    def validate_structure(self) -> "_PopulationCheckpointHead":
        for field_name in (
            "owner_fingerprint",
            "checkpoint_namespace_fingerprint",
            "gate_id_fingerprint",
            "state_fingerprint",
            "state_artifact_sha256",
        ):
            if not _valid_sha256(getattr(self, field_name)):
                raise ValueError("%s must be a sha256 value" % field_name)
        return self


class _PopulationStateEnvelope(_StrictFrozenModel):
    envelope_version: str = "population_gate_state_envelope.v1"
    owner_fingerprint: str
    checkpoint_namespace_fingerprint: str
    gate_id_fingerprint: str
    state: PopulationGateState
    attestation_artifact_sha256: dict[str, str]


class _PopulationAttestationEnvelope(_StrictFrozenModel):
    envelope_version: str = "population_gate_attestation_envelope.v1"
    owner_fingerprint: str
    checkpoint_namespace_fingerprint: str
    gate_id_fingerprint: str
    stage: PopulationVerificationStage
    attestation: PopulationVerificationAttestation


def _checkpoint_head_fingerprint(head: _PopulationCheckpointHead) -> str:
    payload = head.model_dump(by_alias=True, mode="json")
    payload["headFingerprint"] = ""
    return _stable_fingerprint(payload)


class GroundedWorkspacePopulationGateStateStore(PopulationGateStateStore):
    """Filesystem-backed immutable state chain with process-safe revision CAS."""

    _BASE_COMPONENTS = ("checkpoints", "population_gate_v1")
    _HEAD_NAME = "head.json"
    _LOCK_NAME = ".gate.lock"
    _MAX_HEAD_BYTES = 64 * 1024
    _MAX_STATE_BYTES = 8 * 1024 * 1024
    _MAX_ATTESTATION_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        workspace: GroundedContextWorkspace,
        *,
        checkpoint_namespace: str = "online",
    ) -> None:
        if not isinstance(workspace, GroundedContextWorkspace):
            raise ValueError("GroundedContextWorkspace is required")
        namespace = _text(checkpoint_namespace)
        if not namespace:
            raise ValueError("checkpoint_namespace is required")
        supplied_root = Path(workspace.root)
        if supplied_root.is_symlink():
            raise PopulationOnlineGateStorageError("POPULATION_GATE_WORKSPACE_SYMLINK_REJECTED")
        try:
            root = supplied_root.resolve(strict=True)
        except OSError as exc:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_WORKSPACE_INVALID") from exc
        if not root.is_dir():
            raise PopulationOnlineGateStorageError("POPULATION_GATE_WORKSPACE_INVALID")
        owner = _text(workspace.owner_fingerprint)
        if not _valid_sha256(owner):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_OWNER_INVALID")
        self.workspace = workspace
        self.workspace_root = root
        self.owner_fingerprint = owner
        self.checkpoint_namespace_fingerprint = _stable_fingerprint({"checkpointNamespace": namespace})
        descriptor = self._open_directory(
            (
                *self._BASE_COMPONENTS,
                self.checkpoint_namespace_fingerprint,
            ),
            create=True,
        )
        os.close(descriptor)

    @property
    def checkpoint_root(self) -> Path:
        return self.workspace_root.joinpath(
            *self._BASE_COMPONENTS,
            self.checkpoint_namespace_fingerprint,
        )

    def gate_checkpoint_path(self, gate_id: str) -> Path:
        return self.checkpoint_root / self._gate_id_fingerprint(gate_id)

    def load_population_gate(
        self,
        gate_id: str,
    ) -> PopulationGateState | None:
        gate_fingerprint = self._gate_id_fingerprint(gate_id)
        try:
            descriptor = self._open_gate_directory(
                gate_fingerprint,
                create=False,
            )
        except FileNotFoundError:
            return None
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock_gate(descriptor)
            return self._load_locked(
                descriptor,
                gate_id=_text(gate_id),
                gate_fingerprint=gate_fingerprint,
            )
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def create_population_gate(self, state: PopulationGateState) -> bool:
        self._validate_new_state(state, expected_revision=1)
        self._validate_initial_state(state)
        gate_fingerprint = self._gate_id_fingerprint(state.gate_id)
        descriptor = self._open_gate_directory(
            gate_fingerprint,
            create=True,
        )
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock_gate(descriptor)
            current = self._load_locked(
                descriptor,
                gate_id=state.gate_id,
                gate_fingerprint=gate_fingerprint,
            )
            if current is not None:
                return False
            self._publish_state_locked(
                descriptor,
                state,
                gate_fingerprint=gate_fingerprint,
            )
            return True
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def compare_and_swap_population_gate(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        expected_state_fingerprint: str,
        next_state: PopulationGateState,
    ) -> bool:
        normalized_gate_id = _text(gate_id)
        if next_state.gate_id != normalized_gate_id:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NEXT_STATE_BINDING_MISMATCH")
        self._validate_new_state(
            next_state,
            expected_revision=int(expected_revision) + 1,
        )
        gate_fingerprint = self._gate_id_fingerprint(normalized_gate_id)
        try:
            descriptor = self._open_gate_directory(
                gate_fingerprint,
                create=False,
            )
        except FileNotFoundError:
            return False
        lock_descriptor = -1
        try:
            lock_descriptor = self._lock_gate(descriptor)
            current = self._load_locked(
                descriptor,
                gate_id=normalized_gate_id,
                gate_fingerprint=gate_fingerprint,
            )
            if (
                current is None
                or current.revision != int(expected_revision)
                or current.state_fingerprint != _text(expected_state_fingerprint)
            ):
                return False
            self._validate_state_transition(current, next_state)
            self._publish_state_locked(
                descriptor,
                next_state,
                gate_fingerprint=gate_fingerprint,
            )
            return True
        finally:
            if lock_descriptor >= 0:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
                os.close(lock_descriptor)
            os.close(descriptor)

    def _open_gate_directory(
        self,
        gate_fingerprint: str,
        *,
        create: bool,
    ) -> int:
        return self._open_directory(
            (
                *self._BASE_COMPONENTS,
                self.checkpoint_namespace_fingerprint,
                gate_fingerprint,
            ),
            create=create,
        )

    def _open_directory(
        self,
        components: Sequence[str],
        *,
        create: bool,
    ) -> int:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(self.workspace_root, flags)
        try:
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_WORKSPACE_INVALID")
            for component in components:
                if not component or component in {".", ".."} or "/" in component or "\\" in component:
                    raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_COMPONENT_INVALID")
                if create:
                    try:
                        os.mkdir(
                            component,
                            mode=0o700,
                            dir_fd=descriptor,
                        )
                    except FileExistsError:
                        pass
                child = os.open(component, flags, dir_fd=descriptor)
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    os.close(child)
                    raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_DIRECTORY_INVALID")
                os.close(descriptor)
                descriptor = child
            return descriptor
        except Exception:
            os.close(descriptor)
            raise

    def _lock_gate(self, descriptor: int) -> int:
        try:
            lock_descriptor = os.open(
                self._LOCK_NAME,
                os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=descriptor,
            )
        except OSError as exc:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_LOCK_INVALID") from exc
        if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
            os.close(lock_descriptor)
            raise PopulationOnlineGateStorageError("POPULATION_GATE_LOCK_INVALID")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        return lock_descriptor

    def _load_locked(
        self,
        descriptor: int,
        *,
        gate_id: str,
        gate_fingerprint: str,
    ) -> PopulationGateState | None:
        try:
            encoded_head = self._read_regular_at(
                descriptor,
                self._HEAD_NAME,
                max_bytes=self._MAX_HEAD_BYTES,
            )
        except FileNotFoundError:
            try:
                retained_names = {name for name in os.listdir(descriptor) if name != self._LOCK_NAME}
            except OSError as exc:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_LIST_FAILED") from exc
            if retained_names:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_HEAD_MISSING_WITH_HISTORY")
            return None
        try:
            head = _PopulationCheckpointHead.model_validate_json(encoded_head)
        except Exception as exc:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_HEAD_INVALID") from exc
        if (
            head.owner_fingerprint != self.owner_fingerprint
            or head.checkpoint_namespace_fingerprint != self.checkpoint_namespace_fingerprint
            or head.gate_id_fingerprint != gate_fingerprint
            or head.head_fingerprint != _checkpoint_head_fingerprint(head)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_HEAD_BINDING_INVALID")
        state_name = self._state_artifact_name(
            head.revision,
            head.state_fingerprint,
        )
        encoded_state = self._read_immutable_at(
            descriptor,
            state_name,
            expected_sha256=head.state_artifact_sha256,
            max_bytes=self._MAX_STATE_BYTES,
        )
        try:
            envelope = _PopulationStateEnvelope.model_validate_json(encoded_state)
        except Exception as exc:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_STATE_ARTIFACT_INVALID") from exc
        state = envelope.state
        if (
            envelope.owner_fingerprint != self.owner_fingerprint
            or envelope.checkpoint_namespace_fingerprint != self.checkpoint_namespace_fingerprint
            or envelope.gate_id_fingerprint != gate_fingerprint
            or state.gate_id != gate_id
            or state.revision != head.revision
            or state.state_fingerprint != head.state_fingerprint
            or state.state_fingerprint != population_gate_state_fingerprint(state)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_STATE_BINDING_INVALID")
        expected_attestations = self._state_attestations(state)
        if set(envelope.attestation_artifact_sha256) != set(expected_attestations):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_ATTESTATION_SET_INVALID")
        for stage, attestation in expected_attestations.items():
            self._validate_attestation(attestation, expected_stage=stage)
            attestation_envelope = self._attestation_envelope(
                gate_fingerprint,
                stage,
                attestation,
            )
            expected_bytes = _stable_json_bytes(attestation_envelope)
            expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()
            if envelope.attestation_artifact_sha256.get(stage) != expected_sha256:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_ATTESTATION_BINDING_INVALID")
            observed = self._read_immutable_at(
                descriptor,
                self._attestation_artifact_name(
                    stage,
                    attestation.attestation_fingerprint,
                ),
                expected_sha256=expected_sha256,
                max_bytes=self._MAX_ATTESTATION_BYTES,
            )
            if observed != expected_bytes:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_ATTESTATION_CONTENT_INVALID")
        return PopulationGateState.model_validate(state.model_dump(by_alias=True, mode="json"))

    def _publish_state_locked(
        self,
        descriptor: int,
        state: PopulationGateState,
        *,
        gate_fingerprint: str,
    ) -> None:
        attestations = self._state_attestations(state)
        attestation_sha256: dict[str, str] = {}
        for stage, attestation in attestations.items():
            self._validate_attestation(attestation, expected_stage=stage)
            envelope = self._attestation_envelope(
                gate_fingerprint,
                stage,
                attestation,
            )
            encoded = _stable_json_bytes(envelope)
            digest = hashlib.sha256(encoded).hexdigest()
            self._write_immutable_at(
                descriptor,
                self._attestation_artifact_name(
                    stage,
                    attestation.attestation_fingerprint,
                ),
                encoded,
            )
            attestation_sha256[stage] = digest
        state_envelope = _PopulationStateEnvelope(
            owner_fingerprint=self.owner_fingerprint,
            checkpoint_namespace_fingerprint=(self.checkpoint_namespace_fingerprint),
            gate_id_fingerprint=gate_fingerprint,
            state=state,
            attestation_artifact_sha256=attestation_sha256,
        )
        encoded_state = _stable_json_bytes(state_envelope)
        state_sha256 = hashlib.sha256(encoded_state).hexdigest()
        state_name = self._state_artifact_name(
            state.revision,
            state.state_fingerprint,
        )
        self._write_immutable_at(descriptor, state_name, encoded_state)
        pending_head = _PopulationCheckpointHead(
            owner_fingerprint=self.owner_fingerprint,
            checkpoint_namespace_fingerprint=(self.checkpoint_namespace_fingerprint),
            gate_id_fingerprint=gate_fingerprint,
            revision=state.revision,
            state_fingerprint=state.state_fingerprint,
            state_artifact_sha256=state_sha256,
        )
        head = pending_head.model_copy(update={"head_fingerprint": _checkpoint_head_fingerprint(pending_head)})
        self._atomic_replace_at(
            descriptor,
            self._HEAD_NAME,
            _stable_json_bytes(head),
        )

    def _attestation_envelope(
        self,
        gate_fingerprint: str,
        stage: str,
        attestation: PopulationVerificationAttestation,
    ) -> _PopulationAttestationEnvelope:
        return _PopulationAttestationEnvelope(
            owner_fingerprint=self.owner_fingerprint,
            checkpoint_namespace_fingerprint=(self.checkpoint_namespace_fingerprint),
            gate_id_fingerprint=gate_fingerprint,
            stage=PopulationVerificationStage(stage),
            attestation=attestation,
        )

    @staticmethod
    def _state_attestations(
        state: PopulationGateState,
    ) -> dict[str, PopulationVerificationAttestation]:
        retained = {PopulationVerificationStage.GOAL_DECLARATION.value: (state.goal_attestation)}
        if state.pre_execution_attestation is not None:
            retained[PopulationVerificationStage.PRE_EXECUTION.value] = state.pre_execution_attestation
        if state.post_result_attestation is not None:
            retained[PopulationVerificationStage.POST_RESULT.value] = state.post_result_attestation
        return retained

    @staticmethod
    def _validate_attestation(
        attestation: PopulationVerificationAttestation,
        *,
        expected_stage: str,
    ) -> None:
        if (
            _enum_value(attestation.stage) != expected_stage
            or not attestation.attestation_fingerprint
            or attestation.attestation_fingerprint != population_attestation_fingerprint(attestation)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_ATTESTATION_INVALID")

    @staticmethod
    def _validate_new_state(
        state: PopulationGateState,
        *,
        expected_revision: int,
    ) -> None:
        if (
            state.revision != expected_revision
            or not state.state_fingerprint
            or state.state_fingerprint != population_gate_state_fingerprint(state)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NEXT_STATE_INVALID")

    @classmethod
    def _validate_initial_state(cls, state: PopulationGateState) -> None:
        cls._validate_attestation(
            state.goal_attestation,
            expected_stage=(PopulationVerificationStage.GOAL_DECLARATION.value),
        )
        if (
            _enum_value(state.phase) != PopulationGatePhase.GOAL_DECLARATION.value
            or state.graph_fingerprint
            or state.graph_binding is not None
            or state.pre_execution_attestation is not None
            or state.post_result_attestation is not None
            or state.goal_attestation.goal_contract_fingerprint != state.goal_contract_fingerprint
            or not state.goal_attestation.passed
            or not state.goal_attestation.gate_open
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INITIAL_STATE_INVALID")

    @classmethod
    def _validate_state_transition(
        cls,
        current: PopulationGateState,
        next_state: PopulationGateState,
    ) -> None:
        if (
            next_state.gate_id != current.gate_id
            or next_state.goal_contract_fingerprint != current.goal_contract_fingerprint
            or next_state.goal_attestation != current.goal_attestation
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_BINDING_CHANGED")
        if (
            current.graph_receipt is not None
            or next_state.graph_receipt is not None
            or current.node_gate_records
            or next_state.node_gate_records
            or current.retired_node_gate_records
            or next_state.retired_node_gate_records
        ):
            cls._validate_incremental_state_transition(
                current,
                next_state,
            )
            return
        current_phase = _enum_value(current.phase)
        next_phase = _enum_value(next_state.phase)
        if current_phase == PopulationGatePhase.GOAL_DECLARATION.value:
            pre = next_state.pre_execution_attestation
            binding = next_state.graph_binding
            if (
                next_phase != PopulationGatePhase.PRE_EXECUTION.value
                or not next_state.graph_fingerprint
                or binding is None
                or binding.graph_fingerprint != next_state.graph_fingerprint
                or binding.binding_fingerprint != population_execution_graph_binding_fingerprint(binding)
                or pre is None
                or next_state.post_result_attestation is not None
                or current.pre_execution_attestation is not None
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_PRE_TRANSITION_INVALID")
            cls._validate_attestation(
                pre,
                expected_stage=(PopulationVerificationStage.PRE_EXECUTION.value),
            )
            if (
                not pre.passed
                or not pre.gate_open
                or pre.goal_contract_fingerprint != current.goal_contract_fingerprint
                or pre.graph_fingerprint != next_state.graph_fingerprint
                or pre.previous_attestation_fingerprint != current.goal_attestation.attestation_fingerprint
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_PRE_ATTESTATION_CHAIN_INVALID")
            return
        if current_phase == PopulationGatePhase.PRE_EXECUTION.value:
            post = next_state.post_result_attestation
            if (
                next_phase != PopulationGatePhase.POST_RESULT.value
                or next_state.graph_fingerprint != current.graph_fingerprint
                or next_state.graph_binding != current.graph_binding
                or next_state.pre_execution_attestation != current.pre_execution_attestation
                or post is None
                or current.post_result_attestation is not None
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_POST_TRANSITION_INVALID")
            cls._validate_attestation(
                post,
                expected_stage=PopulationVerificationStage.POST_RESULT.value,
            )
            if (
                not post.passed
                or not post.gate_open
                or post.goal_contract_fingerprint != current.goal_contract_fingerprint
                or post.graph_fingerprint != current.graph_fingerprint
                or current.pre_execution_attestation is None
                or post.previous_attestation_fingerprint != current.pre_execution_attestation.attestation_fingerprint
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_POST_ATTESTATION_CHAIN_INVALID")
            return
        raise PopulationOnlineGateStorageError("POPULATION_GATE_TERMINAL_STATE_IMMUTABLE")

    @classmethod
    def _validate_incremental_node_record(
        cls,
        record: Any,
        *,
        receipts: tuple[Any, ...],
        goal_contract_fingerprint: str,
    ) -> None:
        if record.record_fingerprint != population_node_gate_record_fingerprint(record):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_RECORD_INVALID")
        matches = tuple(
            receipt for receipt in receipts if receipt.receipt_fingerprint == record.graph_receipt_fingerprint
        )
        if len(matches) != 1:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_RECEIPT_INVALID")
        receipt = matches[0]
        nodes = {item.query_node_id: item for item in receipt.nodes}
        node = nodes.get(record.query_node_id)
        if (
            node is None
            or record.query_node_id != record.node_binding.query_node_id
            or set(node.consumer_goal_ids) != set(record.node_binding.consumer_goal_ids)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_BINDING_INVALID")
        pre = record.pre_execution_attestation
        cls._validate_attestation(
            pre,
            expected_stage=(PopulationVerificationStage.PRE_EXECUTION.value),
        )
        if (
            not pre.passed
            or not pre.gate_open
            or pre.goal_contract_fingerprint != goal_contract_fingerprint
            or pre.graph_fingerprint != receipt.graph_fingerprint
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_PRE_INVALID")
        post = record.post_result_attestation
        if post is None:
            return
        cls._validate_attestation(
            post,
            expected_stage=(PopulationVerificationStage.POST_RESULT.value),
        )
        if (
            not post.passed
            or not post.gate_open
            or post.goal_contract_fingerprint != goal_contract_fingerprint
            or post.graph_fingerprint != receipt.graph_fingerprint
            or post.previous_attestation_fingerprint != pre.attestation_fingerprint
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_POST_INVALID")

    @classmethod
    def _validate_incremental_state_transition(
        cls,
        current: PopulationGateState,
        next_state: PopulationGateState,
    ) -> None:
        if (
            next_state.graph_binding != current.graph_binding
            or next_state.pre_execution_attestation != current.pre_execution_attestation
            or next_state.post_result_attestation != current.post_result_attestation
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_LEGACY_STATE_CHANGED")
        current_receipt = current.graph_receipt
        next_receipt = next_state.graph_receipt
        if next_receipt is None:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_RECEIPT_REQUIRED")
        if next_receipt.receipt_fingerprint != population_dynamic_graph_receipt_fingerprint(next_receipt):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_RECEIPT_INVALID")
        receipts = tuple(
            [
                *next_state.graph_receipt_history,
                next_receipt,
            ]
        )
        receipt_fingerprints = [item.receipt_fingerprint for item in receipts]
        if len(set(receipt_fingerprints)) != len(receipt_fingerprints):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_RECEIPT_HISTORY_DUPLICATE")
        active_ids = {item.query_node_id for item in next_state.node_gate_records}
        retired_ids = {item.query_node_id for item in next_state.retired_node_gate_records}
        if (
            len(active_ids) != len(next_state.node_gate_records)
            or len(retired_ids) != len(next_state.retired_node_gate_records)
            or active_ids.intersection(retired_ids)
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_RECORD_SET_INVALID")
        for record in (
            *next_state.node_gate_records,
            *next_state.retired_node_gate_records,
        ):
            cls._validate_incremental_node_record(
                record,
                receipts=receipts,
                goal_contract_fingerprint=(next_state.goal_contract_fingerprint),
            )

        if current_receipt is None:
            if (
                next_state.graph_receipt_history != current.graph_receipt_history
                or next_state.graph_revision_evidence_fingerprints != current.graph_revision_evidence_fingerprints
                or next_state.retired_node_gate_records != current.retired_node_gate_records
                or len(next_state.node_gate_records) != 1
                or _enum_value(next_state.phase) != PopulationGatePhase.PRE_EXECUTION.value
                or next_state.graph_fingerprint != next_receipt.graph_fingerprint
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_FIRST_NODE_PRE_INVALID")
            return

        graph_changed = current_receipt.receipt_fingerprint != next_receipt.receipt_fingerprint
        if graph_changed:
            if (
                next_receipt.parent_receipt_fingerprint != current_receipt.receipt_fingerprint
                or next_receipt.graph_version != current_receipt.graph_version + 1
                or next_state.graph_receipt_history
                != tuple(
                    [
                        *current.graph_receipt_history,
                        current_receipt,
                    ]
                )
                or len(next_state.graph_revision_evidence_fingerprints)
                != len(current.graph_revision_evidence_fingerprints) + 1
                or tuple(next_state.graph_revision_evidence_fingerprints[:-1])
                != current.graph_revision_evidence_fingerprints
                or next_state.graph_revision_evidence_fingerprints[-1] != next_receipt.revision_evidence_fingerprint
                or next_state.graph_fingerprint != next_receipt.graph_fingerprint
                or next_state.phase != current.phase
                or next_state.ledger_snapshot_fingerprint != current.ledger_snapshot_fingerprint
                or next_state.published_receipt_fingerprints != current.published_receipt_fingerprints
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_GRAPH_REVISION_INVALID")
            next_active = {item.query_node_id: item for item in next_state.node_gate_records}
            current_active = {item.query_node_id: item for item in current.node_gate_records}
            if any(
                current_active[node_id] != record
                for node_id, record in next_active.items()
                if node_id in current_active
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_CARRIED_RECORD_CHANGED")
            newly_retired = tuple(item for item in current.node_gate_records if item.query_node_id not in next_active)
            if next_state.retired_node_gate_records != tuple(
                [
                    *current.retired_node_gate_records,
                    *newly_retired,
                ]
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_RETIRED_RECORD_SET_INVALID")
            return

        if (
            next_state.graph_receipt_history != current.graph_receipt_history
            or next_state.graph_revision_evidence_fingerprints != current.graph_revision_evidence_fingerprints
            or next_state.retired_node_gate_records != current.retired_node_gate_records
            or next_state.graph_fingerprint != current.graph_fingerprint
        ):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_GRAPH_CHANGED")
        current_records = {item.query_node_id: item for item in current.node_gate_records}
        next_records = {item.query_node_id: item for item in next_state.node_gate_records}
        added_ids = set(next_records) - set(current_records)
        removed_ids = set(current_records) - set(next_records)
        changed_ids = {
            node_id
            for node_id in set(current_records).intersection(next_records)
            if current_records[node_id] != next_records[node_id]
        }
        if removed_ids:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_RECORD_REMOVED")
        if len(added_ids) == 1 and not changed_ids:
            added = next_records[next(iter(added_ids))]
            if (
                added.post_result_attestation is not None
                or _enum_value(next_state.phase) != PopulationGatePhase.PRE_EXECUTION.value
                or next_state.ledger_snapshot_fingerprint != current.ledger_snapshot_fingerprint
                or next_state.published_receipt_fingerprints != current.published_receipt_fingerprints
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_PRE_APPEND_INVALID")
            return
        if not added_ids and len(changed_ids) == 1:
            node_id = next(iter(changed_ids))
            before = current_records[node_id]
            after = next_records[node_id]
            if (
                before.post_result_attestation is not None
                or after.post_result_attestation is None
                or before.model_copy(
                    update={
                        "post_result_attestation": (after.post_result_attestation),
                        "ledger_snapshot_fingerprint": (after.ledger_snapshot_fingerprint),
                        "published_receipt_fingerprints": (after.published_receipt_fingerprints),
                        "record_fingerprint": (after.record_fingerprint),
                    }
                )
                != after
                or _enum_value(next_state.phase) != PopulationGatePhase.POST_RESULT.value
            ):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_NODE_POST_UPDATE_INVALID")
            return
        raise PopulationOnlineGateStorageError("POPULATION_GATE_INCREMENTAL_TRANSITION_INVALID")

    def _gate_id_fingerprint(self, gate_id: str) -> str:
        normalized = _text(gate_id)
        if not normalized:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_ID_REQUIRED")
        return _stable_fingerprint({"gateId": normalized})

    @staticmethod
    def _state_artifact_name(revision: int, fingerprint: str) -> str:
        if int(revision) <= 0 or not _valid_sha256(fingerprint):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_STATE_IDENTITY_INVALID")
        return "state_%020d_%s.json" % (int(revision), fingerprint)

    @staticmethod
    def _attestation_artifact_name(stage: str, fingerprint: str) -> str:
        slugs = {
            PopulationVerificationStage.GOAL_DECLARATION.value: "goal",
            PopulationVerificationStage.PRE_EXECUTION.value: "pre",
            PopulationVerificationStage.POST_RESULT.value: "post",
        }
        slug = slugs.get(stage)
        if slug is None or not _valid_sha256(fingerprint):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_ATTESTATION_IDENTITY_INVALID")
        return "attestation_%s_%s.json" % (slug, fingerprint)

    @classmethod
    def _immutable_marker_name(cls, name: str) -> str:
        return ".immutable_%s.sha256" % name

    @classmethod
    def _read_immutable_at(
        cls,
        descriptor: int,
        name: str,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> bytes:
        if not _valid_sha256(expected_sha256):
            raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_DIGEST_INVALID")
        marker = (
            cls._read_regular_at(
                descriptor,
                cls._immutable_marker_name(name),
                max_bytes=128,
            )
            .decode("ascii")
            .strip()
        )
        if marker != expected_sha256:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_MARKER_MISMATCH")
        encoded = cls._read_regular_at(
            descriptor,
            name,
            max_bytes=max_bytes,
        )
        if hashlib.sha256(encoded).hexdigest() != expected_sha256:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_HASH_MISMATCH")
        return encoded

    @classmethod
    def _write_immutable_at(
        cls,
        descriptor: int,
        name: str,
        encoded: bytes,
    ) -> None:
        digest = hashlib.sha256(encoded).hexdigest()
        try:
            observed = cls._read_regular_at(
                descriptor,
                name,
                max_bytes=max(1, len(encoded)),
            )
        except FileNotFoundError:
            cls._create_regular_at(
                descriptor,
                name,
                encoded,
                mode=0o400,
            )
        else:
            if observed != encoded:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_CONFLICT")
        marker_name = cls._immutable_marker_name(name)
        marker_bytes = digest.encode("ascii")
        try:
            marker = cls._read_regular_at(
                descriptor,
                marker_name,
                max_bytes=128,
            )
        except FileNotFoundError:
            cls._create_regular_at(
                descriptor,
                marker_name,
                marker_bytes,
                mode=0o400,
            )
        else:
            if marker != marker_bytes:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_IMMUTABLE_MARKER_CONFLICT")

    @staticmethod
    def _create_regular_at(
        descriptor: int,
        name: str,
        encoded: bytes,
        *,
        mode: int,
    ) -> None:
        file_descriptor = -1
        try:
            file_descriptor = os.open(
                name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                mode,
                dir_fd=descriptor,
            )
            if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_FILE_INVALID")
            offset = 0
            while offset < len(encoded):
                offset += os.write(file_descriptor, encoded[offset:])
            os.fsync(file_descriptor)
            os.fchmod(file_descriptor, mode)
        except PopulationOnlineGateStorageError:
            raise
        except OSError as exc:
            raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_WRITE_FAILED") from exc
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)

    @staticmethod
    def _read_regular_at(
        descriptor: int,
        name: str,
        *,
        max_bytes: int,
    ) -> bytes:
        file_descriptor = -1
        try:
            file_descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=descriptor,
            )
            metadata = os.fstat(file_descriptor)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_size < 0 or metadata.st_size > max_bytes:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_FILE_INVALID")
            chunks: list[bytes] = []
            remaining = int(metadata.st_size)
            while remaining > 0:
                chunk = os.read(file_descriptor, min(remaining, 1024 * 1024))
                if not chunk:
                    break
                chunks.append(chunk)
                remaining -= len(chunk)
            encoded = b"".join(chunks)
            if len(encoded) != metadata.st_size:
                raise PopulationOnlineGateStorageError("POPULATION_GATE_CHECKPOINT_READ_INCOMPLETE")
            return encoded
        finally:
            if file_descriptor >= 0:
                os.close(file_descriptor)

    @classmethod
    def _atomic_replace_at(
        cls,
        descriptor: int,
        name: str,
        encoded: bytes,
    ) -> None:
        temporary = ".head_%s.tmp" % hashlib.sha256(os.urandom(32)).hexdigest()
        cls._create_regular_at(
            descriptor,
            temporary,
            encoded,
            mode=0o600,
        )
        try:
            os.replace(
                temporary,
                name,
                src_dir_fd=descriptor,
                dst_dir_fd=descriptor,
            )
            os.fsync(descriptor)
        except OSError as exc:
            try:
                os.unlink(temporary, dir_fd=descriptor)
            except OSError:
                pass
            raise PopulationOnlineGateStorageError("POPULATION_GATE_HEAD_COMMIT_FAILED") from exc


def _snapshot_publication_identity(snapshot: Any) -> dict[str, str]:
    return {
        "datasourceFingerprint": _text(getattr(snapshot, "datasource_fingerprint", "")),
        "datasourceEnvironment": _text(getattr(snapshot, "datasource_environment", "")),
        "dataEpoch": _text(getattr(snapshot, "data_epoch", "")),
        "consistencyMode": _text(getattr(snapshot, "consistency_mode", "UNSUPPORTED")) or "UNSUPPORTED",
        "semanticActivationFingerprint": _text(getattr(snapshot, "semantic_activation_fingerprint", "")),
        "cacheGeneration": _text(getattr(snapshot, "cache_generation", "")),
        "capturedAt": _text(getattr(snapshot, "captured_at", "")),
        "unsupportedReason": _text(getattr(snapshot, "unsupported_reason", "")),
    }


def _population_coverage(value: Any) -> PopulationArtifactCoverage:
    normalized = _enum_value(value).upper()
    supported = {
        PopulationArtifactCoverage.ALL_ROWS.value,
        PopulationArtifactCoverage.COMPLETE.value,
        PopulationArtifactCoverage.EXACT_ENTITY_SET.value,
        PopulationArtifactCoverage.TOP_N.value,
        PopulationArtifactCoverage.PREVIEW.value,
        PopulationArtifactCoverage.PARTIAL.value,
    }
    return PopulationArtifactCoverage(
        normalized if normalized in supported else PopulationArtifactCoverage.UNKNOWN.value
    )


class PublishedGroundedPopulationLedgerReader:
    """Project actual verifier-sealed PUBLISHED query ledger items for POST."""

    def __init__(
        self,
        *,
        settings: Any,
        workspace: GroundedContextWorkspace,
        state_store: PopulationGateStateStore,
        ledger_provider: Callable[[], Sequence[GroundedVerifiedQueryArtifact]],
        authority_fingerprint: str,
    ) -> None:
        authority = _text(authority_fingerprint)
        if settings is None:
            raise ValueError("settings are required")
        if not isinstance(workspace, GroundedContextWorkspace):
            raise ValueError("GroundedContextWorkspace is required")
        if not callable(ledger_provider):
            raise ValueError("a dynamic ledger_provider is required")
        if not authority:
            raise ValueError("authority_fingerprint is required")
        try:
            configured_root = Path(settings.resolved_workspace_path).resolve(strict=True)
            workspace_root = Path(workspace.root).resolve(strict=True)
            artifact_root = Path(workspace.artifacts_root).resolve(strict=True)
            workspace_root.relative_to(configured_root)
            artifact_root.relative_to(workspace_root)
        except (AttributeError, OSError, ValueError) as exc:
            raise ValueError("workspace and settings roots do not share one authority") from exc
        self.settings = settings
        self.workspace = workspace
        self.state_store = state_store
        self.ledger_provider = ledger_provider
        self._authority_fingerprint = authority
        self.artifact_store = WorkspaceArtifactStore(
            settings,
            workspace.artifacts_root,
        )

    @property
    def authority_fingerprint(self) -> str:
        return self._authority_fingerprint

    def snapshot_population_artifacts(
        self,
        *,
        gate_id: str,
        goal_contract_fingerprint: str,
        graph_fingerprint: str,
    ) -> PopulationArtifactLedgerSnapshot:
        from merchant_ai.services.grounded_runtime_kernel import (
            GroundedVerifiedQueryArtifact,
            verified_query_artifact_integrity_valid,
        )

        state = self.state_store.load_population_gate(_text(gate_id))
        if state is None:
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_GATE_STATE_NOT_FOUND")
        legacy_pre = state.pre_execution_attestation
        node_records = tuple(state.node_gate_records)
        node_pre_valid = bool(node_records) and all(
            record.record_fingerprint == population_node_gate_record_fingerprint(record)
            and record.pre_execution_attestation.attestation_fingerprint
            == population_attestation_fingerprint(record.pre_execution_attestation)
            for record in node_records
        )
        legacy_pre_valid = bool(
            legacy_pre is not None
            and legacy_pre.attestation_fingerprint == population_attestation_fingerprint(legacy_pre)
        )
        if (
            state.state_fingerprint != population_gate_state_fingerprint(state)
            or state.goal_contract_fingerprint != _text(goal_contract_fingerprint)
            or state.graph_fingerprint != _text(graph_fingerprint)
            or _enum_value(state.phase)
            not in {
                PopulationGatePhase.PRE_EXECUTION.value,
                PopulationGatePhase.POST_RESULT.value,
            }
            or not (legacy_pre_valid or node_pre_valid)
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_GATE_BINDING_INVALID")
        pre_scopes = tuple(
            scope
            for attestation in ([legacy_pre] if legacy_pre_valid else [])
            + [record.pre_execution_attestation for record in node_records if node_pre_valid]
            for scope in attestation.accepted_scopes
        )
        entries: list[PopulationArtifactLedgerEntry] = []
        observed_entry_ids: set[str] = set()
        try:
            raw_ledger = tuple(self.ledger_provider())
        except Exception as exc:
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_PROVIDER_FAILED") from exc
        for artifact in raw_ledger:
            if not isinstance(artifact, GroundedVerifiedQueryArtifact):
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_ARTIFACT_TYPE_INVALID")
            if _text(artifact.publication_status) != "PUBLISHED":
                continue
            if not verified_query_artifact_integrity_valid(artifact):
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_ARTIFACT_INTEGRITY_INVALID")
            if not artifact.verified_evidence.passed:
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_ARTIFACT_NOT_VERIFIED")
            bundle = artifact.run_result.merged_query_bundle
            snapshot = bundle.data_snapshot
            normalized_snapshot_fingerprint = grounded_data_snapshot_fingerprint(snapshot)
            publication_snapshot = _snapshot_publication_identity(snapshot)
            publication_snapshot_fingerprint = _stable_fingerprint(publication_snapshot)
            matching_scopes = tuple(
                scope
                for scope in pre_scopes
                if scope.generation == artifact.generation
                and scope.attempt_id == artifact.attempt_id
                and scope.query_contract_fingerprint == artifact.contract_fingerprint
                and scope.sql_ast_fingerprint == artifact.sql_fingerprint
                and scope.snapshot_fingerprint == normalized_snapshot_fingerprint
            )
            if not matching_scopes:
                continue
            receipts = tuple(artifact.result_artifact_receipts or ())
            if not receipts:
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_PUBLISHED_RECEIPT_REQUIRED")
            for raw_receipt in receipts:
                if not isinstance(raw_receipt, Mapping):
                    raise PopulationOnlineLedgerError("POPULATION_LEDGER_PUBLISHED_RECEIPT_INVALID")
                receipt = dict(raw_receipt)
                manifest = self._validate_published_receipt(
                    artifact,
                    receipt,
                    publication_snapshot=publication_snapshot,
                    publication_snapshot_fingerprint=(publication_snapshot_fingerprint),
                )
                for scope in matching_scopes:
                    if not scope.consumer_goal_id or not scope.query_node_id:
                        raise PopulationOnlineLedgerError("POPULATION_LEDGER_PRE_SCOPE_INVALID")
                    entry_id = "population_result_%s" % _stable_fingerprint(
                        {
                            "queryArtifactId": artifact.artifact_id,
                            "publishedArtifactFingerprint": receipt.get("artifactFingerprint"),
                            "consumerGoalId": scope.consumer_goal_id,
                            "queryNodeId": scope.query_node_id,
                        }
                    )
                    if entry_id in observed_entry_ids:
                        raise PopulationOnlineLedgerError("POPULATION_LEDGER_ENTRY_CONFLICT")
                    observed_entry_ids.add(entry_id)
                    evidence = PopulationArtifactEvidence(
                        artifact_id=_text(receipt.get("artifactFingerprint")),
                        artifact_fingerprint=_text(receipt.get("artifactFingerprint")),
                        artifact_kind=PopulationArtifactKind.QUERY_RESULT,
                        coverage=_population_coverage(manifest.get("resultCoverage")),
                        population_fingerprint=(scope.population_fingerprint),
                        verifier_fingerprint=self.authority_fingerprint,
                        verified=True,
                        immutable=True,
                        goal_contract_fingerprint=(state.goal_contract_fingerprint),
                        graph_fingerprint=state.graph_fingerprint,
                        query_contract_fingerprint=(artifact.contract_fingerprint),
                        sql_ast_fingerprint=artifact.sql_fingerprint,
                        snapshot_fingerprint=(normalized_snapshot_fingerprint),
                        lineage_proof_fingerprints=(scope.proof_fingerprints),
                    )
                    published = seal_population_published_artifact_receipt(
                        PopulationPublishedArtifactReceipt(
                            ledger_artifact_id=entry_id,
                            source_query_artifact_id=artifact.artifact_id,
                            publication_status="PUBLISHED",
                            generation=artifact.generation,
                            attempt_id=artifact.attempt_id,
                            goal_contract_fingerprint=(state.goal_contract_fingerprint),
                            graph_fingerprint=state.graph_fingerprint,
                            query_node_id=scope.query_node_id,
                            covered_consumer_goal_ids=(scope.consumer_goal_id,),
                            result_is_truncated=bool(manifest.get("resultIsTruncated")),
                            stored_row_count=int(manifest.get("storedRowCount") or 0),
                            exact_result_row_count=int(manifest.get("exactResultRowCount") or 0),
                            evidence=evidence,
                        )
                    )
                    entries.append(
                        seal_population_artifact_ledger_entry(
                            PopulationArtifactLedgerEntry(
                                ledger_artifact_id=entry_id,
                                publication_status="PUBLISHED",
                                receipt=published,
                            )
                        )
                    )
            if not verified_query_artifact_integrity_valid(artifact):
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_ARTIFACT_CHANGED_DURING_READ")
        return seal_population_artifact_ledger_snapshot(
            PopulationArtifactLedgerSnapshot(
                ledger_id="population_ledger_%s"
                % _stable_fingerprint(
                    {
                        "ownerFingerprint": (self.workspace.owner_fingerprint),
                        "gateId": gate_id,
                        "authorityFingerprint": self.authority_fingerprint,
                    }
                ),
                ledger_authority_fingerprint=self.authority_fingerprint,
                ledger_revision=state.revision,
                goal_contract_fingerprint=state.goal_contract_fingerprint,
                graph_fingerprint=state.graph_fingerprint,
                entries=tuple(
                    sorted(
                        entries,
                        key=lambda item: item.ledger_artifact_id,
                    )
                ),
            )
        )

    def _validate_published_receipt(
        self,
        artifact: GroundedVerifiedQueryArtifact,
        receipt: Mapping[str, Any],
        *,
        publication_snapshot: Mapping[str, str],
        publication_snapshot_fingerprint: str,
    ) -> dict[str, Any]:
        digest_fields = (
            "artifactFingerprint",
            "queryManifestSha256",
            "rowsSha256",
            "sqlSha256",
            "contractFingerprint",
            "sqlEvidenceFingerprint",
            "dataSnapshotFingerprint",
            "verifiedEvidenceSha256",
            "attemptFingerprint",
        )
        if any(not _valid_sha256(receipt.get(key)) for key in digest_fields):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_RECEIPT_DIGEST_INVALID")
        for address_key, digest_key in (
            ("manifestContentAddress", "queryManifestSha256"),
            ("rowsContentAddress", "rowsSha256"),
            ("sqlContentAddress", "sqlSha256"),
        ):
            if _text(receipt.get(address_key)) != "sha256:%s" % _text(receipt.get(digest_key)):
                raise PopulationOnlineLedgerError("POPULATION_LEDGER_RECEIPT_CONTENT_ADDRESS_INVALID")
        if (
            receipt.get("executionGeneration") != artifact.generation
            or _text(receipt.get("attemptFingerprint"))
            != hashlib.sha256(artifact.attempt_id.encode("utf-8")).hexdigest()
            or _text(receipt.get("contractFingerprint")) != artifact.contract_fingerprint
            or _text(receipt.get("sqlEvidenceFingerprint")) != artifact.sql_fingerprint
            or _text(receipt.get("contextOwnerFingerprint")) != self.workspace.owner_fingerprint
            or _text(receipt.get("dataSnapshotFingerprint")) != publication_snapshot_fingerprint
            or _text(receipt.get("semanticActivationFingerprint"))
            != _text(publication_snapshot.get("semanticActivationFingerprint"))
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_RECEIPT_BINDING_INVALID")
        manifest_path = _safe_relative_path(receipt.get("manifestRelativePath"))
        manifest_result = self.artifact_store.read(
            manifest_path,
            offset=0,
            max_chars=4 * 1024 * 1024,
            require_immutable=True,
        )
        if (
            not manifest_result.get("success")
            or manifest_result.get("truncated")
            or _text(manifest_result.get("sha256")) != _text(receipt.get("queryManifestSha256"))
            or _text(manifest_result.get("contentAddress")) != _text(receipt.get("manifestContentAddress"))
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_MANIFEST_IMMUTABLE_INVALID")
        try:
            manifest = json.loads(_text(manifest_result.get("content")))
        except json.JSONDecodeError as exc:
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_MANIFEST_INVALID") from exc
        if not isinstance(manifest, dict):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_MANIFEST_INVALID")
        scalar_bindings = (
            ("schemaVersion", 2),
            ("artifactKind", "GROUNDED_QUERY_RESULT"),
            ("publicationStatus", "VERIFIED"),
            ("artifactFingerprint", receipt.get("artifactFingerprint")),
            ("executionGeneration", artifact.generation),
            ("executionAttemptId", artifact.attempt_id),
            ("contractFingerprint", artifact.contract_fingerprint),
            ("sqlEvidenceFingerprint", artifact.sql_fingerprint),
            ("sqlSha256", receipt.get("sqlSha256")),
            (
                "contextOwnerFingerprint",
                self.workspace.owner_fingerprint,
            ),
            (
                "semanticActivationFingerprint",
                publication_snapshot.get("semanticActivationFingerprint"),
            ),
        )
        if any(manifest.get(key) != expected for key, expected in scalar_bindings):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_MANIFEST_BINDING_INVALID")
        verified_payload = artifact.verified_evidence.model_dump(
            by_alias=True,
            mode="json",
        )
        verified_sha256 = _stable_fingerprint(verified_payload)
        if (
            manifest.get("verifiedEvidence") != verified_payload
            or _text(manifest.get("verifiedEvidenceSha256")) != verified_sha256
            or _text(receipt.get("verifiedEvidenceSha256")) != verified_sha256
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_VERIFIED_EVIDENCE_BINDING_INVALID")
        if (
            manifest.get("dataSnapshot") != dict(publication_snapshot)
            or _stable_fingerprint(manifest.get("dataSnapshot")) != _text(receipt.get("dataSnapshotFingerprint"))
            or not isinstance(manifest.get("resultIsTruncated"), bool)
            or not isinstance(manifest.get("storedRowCount"), int)
            or isinstance(manifest.get("storedRowCount"), bool)
            or int(manifest.get("storedRowCount")) < 0
            or not isinstance(manifest.get("exactResultRowCount"), int)
            or isinstance(manifest.get("exactResultRowCount"), bool)
            or int(manifest.get("exactResultRowCount")) < 0
            or manifest.get("resultCoverage") != receipt.get("resultCoverage")
            or manifest.get("resultIsTruncated") != receipt.get("resultIsTruncated")
            or manifest.get("storedRowCount") != receipt.get("storedRowCount")
            or manifest.get("exactResultRowCount") != receipt.get("exactResultRowCount")
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_MANIFEST_RESULT_BINDING_INVALID")
        rows_reference = manifest.get("rowsArtifact")
        if not isinstance(rows_reference, Mapping):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_ROWS_REFERENCE_INVALID")
        sql_reference = manifest.get("sqlArtifact")
        if (
            not isinstance(sql_reference, Mapping)
            or _safe_relative_path(sql_reference.get("relativePath"))
            != _safe_relative_path(receipt.get("sqlRelativePath"))
            or _text(sql_reference.get("sha256")) != _text(receipt.get("sqlSha256"))
            or _text(sql_reference.get("contentAddress")) != _text(receipt.get("sqlContentAddress"))
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_SQL_METADATA_BINDING_INVALID")
        rows_path = _safe_relative_path(rows_reference.get("relativePath"))
        if (
            rows_path != _safe_relative_path(receipt.get("rowsRelativePath"))
            or _text(rows_reference.get("sha256")) != _text(receipt.get("rowsSha256"))
            or _text(rows_reference.get("contentAddress")) != _text(receipt.get("rowsContentAddress"))
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_ROWS_BINDING_INVALID")
        rows_result = self.artifact_store.read(
            rows_path,
            offset=0,
            max_chars=1,
            require_immutable=True,
        )
        if (
            not rows_result.get("success")
            or _text(rows_result.get("sha256")) != _text(receipt.get("rowsSha256"))
            or _text(rows_result.get("contentAddress")) != _text(receipt.get("rowsContentAddress"))
        ):
            raise PopulationOnlineLedgerError("POPULATION_LEDGER_ROWS_IMMUTABLE_INVALID")
        return manifest


class PopulationOnlineGateCallResult(_StrictFrozenModel):
    stage: PopulationVerificationStage
    accepted: bool
    code: str
    message: str
    semantic_review: PopulationSemanticReviewerOutcome | None = None
    transition: PopulationGateTransitionResult | None = None


class PopulationOnlineGateFacade:
    """Narrow Goal/PRE/POST facade; prior attestations never cross its API."""

    def __init__(
        self,
        *,
        semantic_reviewer: IndependentPopulationSemanticReviewer,
        coordinator: PopulationGateCoordinator,
        declaration_author_fingerprint: str,
    ) -> None:
        author = _text(declaration_author_fingerprint)
        if not author:
            raise ValueError("declaration_author_fingerprint is required")
        self.semantic_reviewer = semantic_reviewer
        self.coordinator = coordinator
        self.declaration_author_fingerprint = author

    def commit_goal(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        exact_question: str,
        goal_contract: OriginalQuestionGoalContract | Mapping[str, Any] | str,
    ) -> PopulationOnlineGateCallResult:
        try:
            parsed = parse_original_question_goal_contract(goal_contract)
        except Exception as exc:
            return self._failure(
                PopulationVerificationStage.GOAL_DECLARATION,
                "GOAL_CONTRACT_INVALID",
                "The Goal Contract is invalid: %s" % str(exc)[:300],
            )
        review = self.semantic_reviewer.review(
            effective_question=exact_question,
            contract=parsed,
            declaration_author_fingerprint=(self.declaration_author_fingerprint),
        )
        if not review.passed or review.review is None:
            return PopulationOnlineGateCallResult(
                stage=PopulationVerificationStage.GOAL_DECLARATION,
                accepted=False,
                code="SEMANTIC_REVIEW_REJECTED",
                message="Independent population semantic review was rejected.",
                semantic_review=review,
            )
        gate_input = goal_declaration_population_input_from_review(
            parsed,
            review,
            declaration_author_fingerprint=(self.declaration_author_fingerprint),
            trusted_semantic_verifier_fingerprints=(self.coordinator.trusted_semantic_verifier_fingerprints),
        )
        transition = self.coordinator.commit_goal_declaration(
            PopulationGoalDeclarationCommand(
                gate_id=_text(gate_id),
                expected_revision=int(expected_revision),
                goal_contract_fingerprint=(gate_input.goal_contract_fingerprint),
                question_fingerprint=gate_input.question_fingerprint,
                goal_skeleton_fingerprint=(gate_input.goal_skeleton_fingerprint),
                declaration_author_fingerprint=(self.declaration_author_fingerprint),
                semantic_review=gate_input.semantic_review,
                declarations=gate_input.declarations,
            )
        )
        return self._transition_result(
            PopulationVerificationStage.GOAL_DECLARATION,
            transition,
            semantic_review=review,
        )

    def authorize_pre_execution(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        graph_binding: PopulationExecutionGraphBinding,
        claims: Sequence[PopulationExecutionClaim],
    ) -> PopulationOnlineGateCallResult:
        state = self.coordinator.get_state(_text(gate_id))
        if state is None:
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "STATE_NOT_FOUND",
                "The population Goal attestation is unavailable.",
            )
        transition = self.coordinator.authorize_pre_execution(
            PopulationPreExecutionCommand(
                gate_id=state.gate_id,
                expected_revision=int(expected_revision),
                goal_contract_fingerprint=(state.goal_contract_fingerprint),
                graph_binding=graph_binding,
                claims=tuple(claims),
            )
        )
        return self._transition_result(
            PopulationVerificationStage.PRE_EXECUTION,
            transition,
        )

    def authorize_node_pre_execution(
        self,
        command: PopulationNodePreExecutionCommand,
    ) -> PopulationOnlineGateCallResult:
        state = self.coordinator.get_state(_text(command.gate_id))
        if state is None:
            return self._failure(
                PopulationVerificationStage.PRE_EXECUTION,
                "STATE_NOT_FOUND",
                "The population Goal attestation is unavailable.",
            )
        transition = self.coordinator.authorize_node_pre_execution(
            command.model_copy(update={"goal_contract_fingerprint": (state.goal_contract_fingerprint)})
        )
        return self._transition_result(
            PopulationVerificationStage.PRE_EXECUTION,
            transition,
        )

    def commit_post_result(
        self,
        *,
        gate_id: str,
        expected_revision: int,
        selections: Sequence[PopulationResultSelection],
    ) -> PopulationOnlineGateCallResult:
        state = self.coordinator.get_state(_text(gate_id))
        if state is None:
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "STATE_NOT_FOUND",
                "The PRE_EXECUTION attestation is unavailable.",
            )
        transition = self.coordinator.commit_post_result(
            PopulationPostResultCommand(
                gate_id=state.gate_id,
                expected_revision=int(expected_revision),
                goal_contract_fingerprint=(state.goal_contract_fingerprint),
                graph_fingerprint=state.graph_fingerprint,
                selections=tuple(selections),
            )
        )
        return self._transition_result(
            PopulationVerificationStage.POST_RESULT,
            transition,
        )

    def commit_node_post_result(
        self,
        command: PopulationNodePostResultCommand,
    ) -> PopulationOnlineGateCallResult:
        state = self.coordinator.get_state(_text(command.gate_id))
        if state is None:
            return self._failure(
                PopulationVerificationStage.POST_RESULT,
                "STATE_NOT_FOUND",
                "The node PRE attestation is unavailable.",
            )
        transition = self.coordinator.commit_node_post_result(
            command.model_copy(update={"goal_contract_fingerprint": (state.goal_contract_fingerprint)})
        )
        return self._transition_result(
            PopulationVerificationStage.POST_RESULT,
            transition,
        )

    @staticmethod
    def _transition_result(
        stage: PopulationVerificationStage,
        transition: PopulationGateTransitionResult,
        *,
        semantic_review: PopulationSemanticReviewerOutcome | None = None,
    ) -> PopulationOnlineGateCallResult:
        return PopulationOnlineGateCallResult(
            stage=stage,
            accepted=transition.accepted and transition.committed,
            code=_enum_value(transition.code),
            message=transition.message,
            semantic_review=semantic_review,
            transition=transition,
        )

    @staticmethod
    def _failure(
        stage: PopulationVerificationStage,
        code: str,
        message: str,
    ) -> PopulationOnlineGateCallResult:
        return PopulationOnlineGateCallResult(
            stage=stage,
            accepted=False,
            code=code,
            message=message,
        )
