from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Literal, Optional, Sequence

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from pydantic import Field

from merchant_ai.models import APIModel


GroundedSubagentCapability = Literal[
    "READ_CONTEXT",
    "QUERY_BRANCH",
    "RUN_SKILL",
]


class GroundedSubagentBudget(APIModel):
    max_tool_calls: int = Field(default=8, ge=1, le=24)
    timeout_seconds: float = Field(default=45.0, ge=1.0, le=300.0)


class GroundedSubagentEvidenceRequirement(APIModel):
    requirement_id: str
    description: str
    accepted_ref_types: list[str] = Field(default_factory=list)
    required: bool = True


class GroundedSubagentGoalContract(APIModel):
    """One immutable child Goal authored and accepted by the Root Core.

    The worker chooses how to execute the child Goal inside its grant. It may
    propose follow-up Goals, but cannot mutate this contract or its parent Goal
    ledger. A retry is a new generation of the same ``sub_goal_id``.
    """

    sub_goal_id: str
    parent_goal_ids: list[str] = Field(default_factory=list)
    objective: str
    required_outputs: list[str] = Field(default_factory=list)
    input_artifact_refs: list[str] = Field(default_factory=list)
    evidence_requirements: list[GroundedSubagentEvidenceRequirement] = Field(
        default_factory=list
    )
    allowed_capabilities: list[GroundedSubagentCapability] = Field(
        default_factory=list
    )
    budget: GroundedSubagentBudget = Field(default_factory=GroundedSubagentBudget)
    generation: int = Field(default=1, ge=1, le=1000)
    query_branch_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    input_payload: dict[str, Any] = Field(default_factory=dict)

    def contract_payload(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True, mode="json")

    def contract_fingerprint(self) -> str:
        return _stable_fingerprint(self.contract_payload())


# Compatibility name for internal callers written before the protocol was
# promoted from a loose task instruction to an explicit child Goal contract.
GroundedSubagentTaskSpec = GroundedSubagentGoalContract


class GroundedSubagentDispatchPlan(APIModel):
    """A dynamic serial or parallel batch chosen by the single Root Core."""

    tasks: list[GroundedSubagentGoalContract] = Field(default_factory=list)
    parallel: bool = False
    reason: str = ""


class GroundedSubagentCapabilityGrant(APIModel):
    """Server-issued, task-scoped authority mounted into one isolated graph."""

    grant_id: str
    sub_goal_id: str
    parent_goal_ids: list[str] = Field(default_factory=list)
    generation: int = 1
    goal_contract_fingerprint: str
    capabilities: list[GroundedSubagentCapability] = Field(default_factory=list)
    allowed_tool_names: list[str] = Field(default_factory=list)
    query_branch_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    skill_names: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(default=1, ge=1, le=24)
    output_authority: Literal["ADVISORY"] = "ADVISORY"
    grant_fingerprint: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "grantId": self.grant_id,
            "subGoalId": self.sub_goal_id,
            "parentGoalIds": list(dict.fromkeys(self.parent_goal_ids)),
            "generation": int(self.generation),
            "goalContractFingerprint": self.goal_contract_fingerprint,
            "capabilities": sorted(set(self.capabilities)),
            "allowedToolNames": sorted(set(self.allowed_tool_names)),
            "queryBranchIds": list(dict.fromkeys(self.query_branch_ids)),
            "artifactIds": list(dict.fromkeys(self.artifact_ids)),
            "skillNames": list(dict.fromkeys(self.skill_names)),
            "maxToolCalls": int(self.max_tool_calls),
            "outputAuthority": self.output_authority,
        }

    def with_fingerprint(self) -> "GroundedSubagentCapabilityGrant":
        return self.model_copy(
            update={
                "grant_fingerprint": _stable_fingerprint(
                    self.canonical_payload()
                )
            }
        )

    def fingerprint_valid(self) -> bool:
        return bool(self.grant_fingerprint) and self.grant_fingerprint == (
            _stable_fingerprint(self.canonical_payload())
        )


class GroundedSubagentTaskOutcome(APIModel):
    sub_goal_id: str
    generation: int
    status: Literal["COMPLETED", "FAILED"]
    grant: GroundedSubagentCapabilityGrant
    checkpoint: dict[str, Any] = Field(default_factory=dict)
    raw_output: str = ""
    advisory_output: dict[str, Any] = Field(default_factory=dict)
    validation_errors: list[str] = Field(default_factory=list)
    update_count: int = 0
    error: str = ""


@dataclass(frozen=True)
class PreparedIsolatedSubagentTask:
    """A validated job plus the scope needed to execute it safely."""

    task: GroundedSubagentGoalContract
    grant: GroundedSubagentCapabilityGrant
    job: "IsolatedSubagentJob"
    runner: Callable[["IsolatedSubagentJob"], "IsolatedSubagentResult"]


class GroundedSubagentCapabilityMiddleware(AgentMiddleware):
    """Fail closed for every tool not minted into this task's grant."""

    name = "GroundedSubagentCapabilityMiddleware"

    def __init__(self, grant: GroundedSubagentCapabilityGrant) -> None:
        if not grant.fingerprint_valid():
            raise RuntimeError("isolated subagent capability grant is invalid")
        self.grant = grant
        self._allowed = frozenset(grant.allowed_tool_names)
        self._tool_calls = 0
        self._lock = RLock()

    @property
    def tool_calls(self) -> int:
        with self._lock:
            return self._tool_calls

    def wrap_tool_call(
        self,
        request: Any,
        handler: Callable[[Any], Any],
    ) -> Any:
        tool_call = dict(getattr(request, "tool_call", None) or {})
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        if tool_name not in self._allowed:
            return ToolMessage(
                content=json.dumps(
                    {
                        "status": "CAPABILITY_DENIED",
                        "code": "SUBAGENT_TOOL_NOT_GRANTED",
                        "tool": tool_name,
                        "grantId": self.grant.grant_id,
                    },
                    ensure_ascii=False,
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )
        with self._lock:
            if self._tool_calls >= self.grant.max_tool_calls:
                return ToolMessage(
                    content=json.dumps(
                        {
                            "status": "CAPABILITY_DENIED",
                            "code": "SUBAGENT_TOOL_CALL_BUDGET_EXHAUSTED",
                            "grantId": self.grant.grant_id,
                            "maxToolCalls": self.grant.max_tool_calls,
                        },
                        ensure_ascii=False,
                    ),
                    name=tool_name,
                    tool_call_id=tool_call_id,
                    status="error",
                )
            self._tool_calls += 1
        return handler(request)


@dataclass(frozen=True)
class IsolatedSubagentJob:
    job_id: str
    thread_id: str
    system_prompt: str
    user_payload: dict[str, Any]
    backend: Any
    tools: list[Any] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    middleware: list[Any] = field(default_factory=list)
    permissions: list[Any] = field(default_factory=list)
    subagents: list[dict[str, Any]] = field(default_factory=list)
    model_timeout_seconds: Optional[float] = None
    capability_grant: Optional[GroundedSubagentCapabilityGrant] = None


@dataclass(frozen=True)
class IsolatedSubagentResult:
    job_id: str
    thread_id: str
    checkpoint: dict[str, Any]
    raw_output: str
    update_count: int


class IsolatedSubagentRuntime:
    """Generic isolation Harness used by Skills and other long sub-processes.

    It deliberately knows nothing about Skill semantics, SQL, Topic routing, or
    answer composition.  Callers mount the resources and tools needed by one
    job.  This keeps subagent runtime separate from agent/skill definitions.
    """

    def __init__(
        self,
        *,
        model: Any,
        agent_factory: Any,
        checkpointer: Any,
        checkpoint_config_factory: Optional[
            Callable[[str, str], dict[str, Any]]
        ] = None,
    ) -> None:
        self.model = model
        self.agent_factory = agent_factory
        self.checkpointer = checkpointer
        self.checkpoint_config_factory = checkpoint_config_factory

    def run(
        self,
        job: IsolatedSubagentJob,
        *,
        on_progress: Optional[Callable[[str, str, str], None]] = None,
    ) -> IsolatedSubagentResult:
        if self.checkpointer is None:
            raise RuntimeError("isolated subagent requires a checkpoint backend")
        model = self.model
        if job.model_timeout_seconds is not None:
            timeout_seconds = float(job.model_timeout_seconds)
            if timeout_seconds <= 0:
                raise RuntimeError("isolated subagent model timeout must be positive")
            bind = getattr(model, "bind", None)
            if not callable(bind):
                raise RuntimeError(
                    "isolated subagent model cannot enforce the requested provider timeout"
                )
            model = bind(timeout=timeout_seconds)
        middleware = list(job.middleware)
        grant = job.capability_grant
        if grant is not None:
            if not grant.fingerprint_valid():
                raise RuntimeError(
                    "isolated subagent capability grant fingerprint mismatch"
                )
            custom_tool_names = {
                _tool_name(item) for item in job.tools if _tool_name(item)
            }
            ungranted_custom_tools = sorted(
                custom_tool_names - set(grant.allowed_tool_names)
            )
            if ungranted_custom_tools:
                raise RuntimeError(
                    "isolated subagent job mounts ungranted tools: %s"
                    % ",".join(ungranted_custom_tools)
                )
            if job.skills:
                raise RuntimeError(
                    "capability-scoped jobs mount Skills through an explicit backend"
                )
            middleware.append(GroundedSubagentCapabilityMiddleware(grant))
        graph = self.agent_factory(
            model=model,
            tools=list(job.tools),
            system_prompt=job.system_prompt,
            middleware=middleware,
            subagents=list(job.subagents),
            skills=list(job.skills) or None,
            permissions=list(job.permissions),
            backend=job.backend,
            checkpointer=self.checkpointer,
            name="isolated_worker_%s" % _safe_agent_name_segment(job.job_id),
        )
        config = (
            self.checkpoint_config_factory(job.thread_id, job.job_id)
            if self.checkpoint_config_factory is not None
            else {
                "configurable": {
                    "thread_id": job.thread_id,
                    "run_id": job.job_id,
                }
            }
        )
        checkpoint = {
            "threadId": job.thread_id,
            "runId": job.job_id,
            "checkpointNamespace": str(
                (config.get("configurable") or {}).get("checkpoint_ns") or ""
            ),
        }
        if on_progress is not None:
            on_progress("subagent", "started", job.job_id)
        update_count = 0
        for update in graph.stream(
            {"messages": [{"role": "user", "content": _json_text(job.user_payload)}]},
            config=config,
            stream_mode="updates",
        ):
            update_count += 1
            if on_progress is not None and update_count <= 32:
                detail = (
                    ",".join(str(key) for key in update.keys())
                    if isinstance(update, dict)
                    else type(update).__name__
                )
                on_progress("subagent_step", "running", detail)
        snapshot = graph.get_state(config)
        messages = list((getattr(snapshot, "values", {}) or {}).get("messages") or [])
        raw_output = _last_assistant_text(messages)
        if on_progress is not None:
            on_progress("subagent", "completed", "updates=%d" % update_count)
        return IsolatedSubagentResult(
            job_id=job.job_id,
            thread_id=job.thread_id,
            checkpoint=checkpoint,
            raw_output=raw_output,
            update_count=update_count,
        )


def issue_grounded_subagent_capability_grant(
    task: GroundedSubagentGoalContract,
    *,
    allowed_tool_names: Sequence[str],
    query_branch_ids: Sequence[str] = (),
    artifact_ids: Sequence[str] = (),
    skill_names: Sequence[str] = (),
) -> GroundedSubagentCapabilityGrant:
    """Mint one immutable grant after the Root has validated current state."""

    sub_goal_id = _safe_identifier(
        task.sub_goal_id,
        field_name="sub_goal_id",
    )
    capabilities = list(dict.fromkeys(task.allowed_capabilities))
    if not capabilities:
        raise ValueError("subagent task requires at least one capability")
    grant = GroundedSubagentCapabilityGrant(
        grant_id="grant_%s_%s"
        % (
            sub_goal_id[:40],
            hashlib.sha256(
                json.dumps(
                    {
                        "task": task.model_dump(by_alias=True, mode="json"),
                        "tools": sorted(set(allowed_tool_names)),
                        "branches": list(query_branch_ids),
                        "artifacts": list(artifact_ids),
                        "skills": list(skill_names),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()[:16],
        ),
        sub_goal_id=sub_goal_id,
        parent_goal_ids=list(dict.fromkeys(task.parent_goal_ids)),
        generation=task.generation,
        goal_contract_fingerprint=task.contract_fingerprint(),
        capabilities=capabilities,
        allowed_tool_names=sorted(
            set(str(item or "").strip() for item in allowed_tool_names if str(item or "").strip())
        ),
        query_branch_ids=list(dict.fromkeys(str(item) for item in query_branch_ids)),
        artifact_ids=list(dict.fromkeys(str(item) for item in artifact_ids)),
        skill_names=list(dict.fromkeys(str(item) for item in skill_names)),
        max_tool_calls=task.budget.max_tool_calls,
    )
    if not grant.allowed_tool_names:
        raise ValueError("subagent capability grant requires at least one tool")
    return grant.with_fingerprint()


def dispatch_prepared_subagent_tasks(
    tasks: Sequence[PreparedIsolatedSubagentTask],
    *,
    parallel: bool,
    max_workers: int,
) -> list[GroundedSubagentTaskOutcome]:
    """Execute isolated jobs serially or in parallel, preserving input order."""

    prepared = list(tasks)
    if not prepared:
        return []
    task_generations = [
        (item.task.sub_goal_id, item.task.generation) for item in prepared
    ]
    if len(set(task_generations)) != len(task_generations):
        raise ValueError(
            "subagent Goal generations must be unique within one dispatch"
        )

    def invoke(item: PreparedIsolatedSubagentTask) -> GroundedSubagentTaskOutcome:
        try:
            result = item.runner(item.job)
        except Exception as exc:
            return GroundedSubagentTaskOutcome(
                sub_goal_id=item.task.sub_goal_id,
                generation=item.task.generation,
                status="FAILED",
                grant=item.grant,
                error="%s:%s" % (type(exc).__name__, str(exc)[:500]),
            )
        raw_output = str(result.raw_output or "")[:40_000]
        advisory_output, validation_errors = _validate_advisory_output(
            raw_output,
            item.task,
        )
        return GroundedSubagentTaskOutcome(
            sub_goal_id=item.task.sub_goal_id,
            generation=item.task.generation,
            status="FAILED" if validation_errors else "COMPLETED",
            grant=item.grant,
            checkpoint=dict(result.checkpoint),
            raw_output=raw_output,
            advisory_output=advisory_output,
            validation_errors=validation_errors,
            update_count=max(0, int(result.update_count or 0)),
            error=(
                "SUBAGENT_OUTPUT_CONTRACT_REJECTED"
                if validation_errors
                else ""
            ),
        )

    if not parallel or len(prepared) == 1:
        return [invoke(item) for item in prepared]

    workers = max(1, min(int(max_workers or 1), len(prepared), 8))
    indexed: dict[int, GroundedSubagentTaskOutcome] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_by_index = {
            pool.submit(invoke, item): index
            for index, item in enumerate(prepared)
        }
        for future in as_completed(future_by_index):
            indexed[future_by_index[future]] = future.result()
    return [indexed[index] for index in range(len(prepared))]


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _validate_advisory_output(
    raw_output: str,
    contract: GroundedSubagentGoalContract,
) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(str(raw_output or ""))
    except (TypeError, ValueError):
        return {}, ["SUBAGENT_OUTPUT_NOT_JSON"]
    if not isinstance(payload, dict):
        return {}, ["SUBAGENT_OUTPUT_NOT_OBJECT"]
    forbidden = {
        "finalAnswer",
        "answer",
        "publishEvidence",
        "goalMutation",
        "contractMutation",
        "verifiedEvidence",
    }
    errors = [
        "SUBAGENT_OUTPUT_FORBIDDEN_FIELD:%s" % key
        for key in sorted(forbidden.intersection(payload))
    ]
    required = list(
        dict.fromkeys(
            [
                *contract.required_outputs,
                "summary",
                "evidenceRefs",
                "gaps",
                "recommendedNextAction",
                "proposedSubGoals",
                "evidenceGaps",
            ]
        )
    )
    errors.extend(
        "SUBAGENT_OUTPUT_REQUIRED_FIELD_MISSING:%s" % key
        for key in required
        if key not in payload
    )
    for key in ("evidenceRefs", "gaps", "proposedSubGoals", "evidenceGaps"):
        if key in payload and not isinstance(payload[key], list):
            errors.append("SUBAGENT_OUTPUT_FIELD_NOT_LIST:%s" % key)
    if payload.get("subGoalId") not in (None, "", contract.sub_goal_id):
        errors.append("SUBAGENT_OUTPUT_SUB_GOAL_MISMATCH")
    if payload.get("generation") not in (None, "", contract.generation):
        errors.append("SUBAGENT_OUTPUT_GENERATION_MISMATCH")
    # The worker may propose another child Goal, but proposals carry no tools,
    # generation authority, or executable grant. The Root must author a fresh
    # GroundedSubagentGoalContract before anything can run.
    for proposal in payload.get("proposedSubGoals") or []:
        if not isinstance(proposal, dict):
            errors.append("SUBAGENT_PROPOSED_SUB_GOAL_NOT_OBJECT")
            continue
        if any(
            key in proposal
            for key in (
                "allowedCapabilities",
                "capabilityGrant",
                "tools",
                "generation",
                "publishEvidence",
            )
        ):
            errors.append("SUBAGENT_PROPOSED_SUB_GOAL_EXECUTABLE_FIELD_DENIED")
    return dict(payload), list(dict.fromkeys(errors))


def _stable_fingerprint(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _safe_identifier(value: str, *, field_name: str) -> str:
    token = str(value or "").strip()
    if (
        not token
        or len(token) > 96
        or any(
            not (
                character.isascii()
                and (character.isalnum() or character in {"_", "-", "."})
            )
            for character in token
        )
    ):
        raise ValueError("invalid isolated subagent %s" % field_name)
    return token


def _tool_name(item: Any) -> str:
    if isinstance(item, dict):
        function = item.get("function") if isinstance(item.get("function"), dict) else {}
        return str(item.get("name") or function.get("name") or "")
    return str(getattr(item, "name", "") or "")


def _safe_agent_name_segment(value: str) -> str:
    normalized = "".join(
        character
        if (
            "a" <= character <= "z"
            or "A" <= character <= "Z"
            or "0" <= character <= "9"
            or character == "_"
        )
        else "_"
        for character in str(value or "")
    )[:48]
    return normalized or "job"


def _last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if str(getattr(message, "type", "") or "") not in {"ai", "assistant"}:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part).strip()
    return ""
