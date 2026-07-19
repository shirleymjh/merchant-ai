from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Iterator, Optional

from pydantic import Field

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_query_contract import GroundedBindingHints
from merchant_ai.services.grounded_runtime_budget import GroundedRuntimeBudget
from merchant_ai.services.grounded_runtime_kernel import GroundedRuntimeSession


class GroundedQueryBranchSpec(APIModel):
    """One Core-declared query scope created before semantic retrieval.

    The declaration groups original-question goals and limits the Topic
    workspace.  It deliberately does not select SQL, impose a fixed stage
    graph, or authorize semantic bindings.
    """

    query_id: str
    objective: str = ""
    goal_ids: list[str] = Field(default_factory=list)
    topic_scope: list[str] = Field(default_factory=list)


class GroundedBranchPrepareSpec(APIModel):
    """Exact semantic coordinates and bindings for one declared branch."""

    query_id: str
    semantic_paths: list[str] = Field(default_factory=list)
    read_ref_ids: list[str] = Field(default_factory=list)
    binding_hints: GroundedBindingHints = Field(
        default_factory=GroundedBindingHints
    )
    # Retained for the V1 tool contract.  In V2 this must be empty or exactly
    # match the immutable branch declaration.
    goal_ids: list[str] = Field(default_factory=list)


class GroundedBranchBudgetExceeded(RuntimeError):
    """Raised when one branch exhausts its local lease, not the whole run."""

    def __init__(self, code: str, report: dict[str, Any]):
        self.code = str(code or "BRANCH_BUDGET_EXCEEDED")
        self.report = dict(report)
        super().__init__(self.code)


@dataclass(frozen=True)
class GroundedBranchBudgetLimits:
    max_semantic_reads: int = 8
    max_semantic_chars: int = 200_000
    max_contract_attempts: int = 3
    max_doris_queries: int = 2
    max_duration_seconds: float = 50.0
    finalization_reserve_seconds: float = 15.0

    @classmethod
    def from_settings(cls, settings: Any) -> "GroundedBranchBudgetLimits":
        return cls(
            max_semantic_reads=max(
                1,
                int(
                    getattr(
                        settings,
                        "grounded_branch_max_semantic_reads",
                        8,
                    )
                    or 8
                ),
            ),
            max_semantic_chars=max(
                1_000,
                int(
                    getattr(
                        settings,
                        "grounded_branch_max_semantic_chars",
                        200_000,
                    )
                    or 200_000
                ),
            ),
            max_contract_attempts=max(
                1,
                int(
                    getattr(
                        settings,
                        "grounded_branch_max_contract_attempts",
                        3,
                    )
                    or 3
                ),
            ),
            max_doris_queries=max(
                1,
                int(
                    getattr(
                        settings,
                        "grounded_branch_max_doris_queries",
                        2,
                    )
                    or 2
                ),
            ),
            max_duration_seconds=max(
                1.0,
                float(
                    getattr(
                        settings,
                        "grounded_branch_max_duration_seconds",
                        50,
                    )
                    or 50
                ),
            ),
            finalization_reserve_seconds=max(
                0.0,
                float(
                    getattr(
                        settings,
                        "grounded_finalization_reserve_seconds",
                        15,
                    )
                    or 15
                ),
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "maxSemanticReads": self.max_semantic_reads,
            "maxSemanticChars": self.max_semantic_chars,
            "maxContractAttempts": self.max_contract_attempts,
            "maxDorisQueries": self.max_doris_queries,
            "maxDurationSeconds": self.max_duration_seconds,
            "finalizationReserveSeconds": self.finalization_reserve_seconds,
        }


class GroundedBranchBudget:
    """Local branch lease backed by the one thread-safe run budget.

    Reads and Doris calls are charged to both the branch lease and the parent
    run.  Contract attempts are local deterministic work and receive timing
    telemetry without pretending that the single Core made a branch-local LLM
    call.
    """

    def __init__(
        self,
        branch_id: str,
        limits: GroundedBranchBudgetLimits,
        *,
        parent: Optional[GroundedRuntimeBudget] = None,
    ) -> None:
        self.branch_id = str(branch_id or "branch")
        self.limits = limits
        self.parent = parent
        self._started = time.monotonic()
        self._lock = RLock()
        self._semantic_reads = 0
        self._semantic_chars = 0
        self._contract_attempts = 0
        self._doris_queries = 0
        self._stage_calls: dict[str, int] = {}
        self._stage_duration_seconds: dict[str, float] = {}
        self._denied_attempts: list[dict[str, Any]] = []

    def _elapsed_seconds_locked(self) -> float:
        return max(0.0, time.monotonic() - self._started)

    def _deny(self, code: str, operation: str) -> None:
        report = self.report()
        with self._lock:
            self._denied_attempts.append(
                {
                    "code": code,
                    "operation": operation,
                    "elapsedMs": round(
                        self._elapsed_seconds_locked() * 1000,
                        3,
                    ),
                }
            )
            self._denied_attempts = self._denied_attempts[-16:]
            report = self._report_locked()
        raise GroundedBranchBudgetExceeded(code, report)

    def _check_time(self, operation: str) -> None:
        with self._lock:
            if self._elapsed_seconds_locked() >= self.limits.max_duration_seconds:
                code = "BRANCH_DURATION_EXHAUSTED"
            else:
                code = ""
        if code:
            self._deny(code, operation)
        if (
            self.parent is not None
            and self.parent.remaining_seconds()
            <= self.limits.finalization_reserve_seconds
        ):
            self._deny("FINALIZATION_RESERVE_REACHED", operation)

    def consume_semantic_read(self, *, path: str, content_chars: int) -> None:
        self._check_time("semantic_read:%s" % path)
        normalized_chars = max(0, int(content_chars or 0))
        with self._lock:
            if self._semantic_reads + 1 > self.limits.max_semantic_reads:
                code = "BRANCH_SEMANTIC_READ_LIMIT"
            elif self._semantic_chars + normalized_chars > self.limits.max_semantic_chars:
                code = "BRANCH_SEMANTIC_CHAR_LIMIT"
            else:
                code = ""
        if code:
            self._deny(code, "semantic_read:%s" % path)
        if self.parent is not None:
            self.parent.consume_tool_call(
                "branch.%s.semantic_read" % self.branch_id
            )
        with self._lock:
            self._semantic_reads += 1
            self._semantic_chars += normalized_chars

    def consume_contract_attempt(self) -> None:
        self._check_time("contract")
        with self._lock:
            if self._contract_attempts + 1 > self.limits.max_contract_attempts:
                code = "BRANCH_CONTRACT_ATTEMPT_LIMIT"
            else:
                code = ""
        if code:
            self._deny(code, "contract")
        with self._lock:
            self._contract_attempts += 1

    def consume_doris_query(self) -> None:
        self._check_time("doris")
        with self._lock:
            if self._doris_queries + 1 > self.limits.max_doris_queries:
                code = "BRANCH_DORIS_QUERY_LIMIT"
            else:
                code = ""
        if code:
            self._deny(code, "doris")
        if self.parent is not None:
            self.parent.consume_doris_query(
                name="parallel.%s" % self.branch_id
            )
        with self._lock:
            self._doris_queries += 1

    @contextmanager
    def stage(self, name: str) -> Iterator["GroundedBranchBudget"]:
        stage_name = str(name or "unknown")
        started = time.monotonic()
        parent_stage = (
            self.parent.stage("branch.%s.%s" % (self.branch_id, stage_name))
            if self.parent is not None
            else None
        )
        try:
            if parent_stage is None:
                yield self
            else:
                with parent_stage:
                    yield self
        finally:
            duration = max(0.0, time.monotonic() - started)
            with self._lock:
                self._stage_calls[stage_name] = (
                    self._stage_calls.get(stage_name, 0) + 1
                )
                self._stage_duration_seconds[stage_name] = (
                    self._stage_duration_seconds.get(stage_name, 0.0)
                    + duration
                )

    def report(self) -> dict[str, Any]:
        with self._lock:
            return self._report_locked()

    def _report_locked(self) -> dict[str, Any]:
        return {
            "branchId": self.branch_id,
            "elapsedMs": round(self._elapsed_seconds_locked() * 1000, 3),
            "limits": self.limits.as_dict(),
            "usage": {
                "semanticReads": self._semantic_reads,
                "semanticChars": self._semantic_chars,
                "contractAttempts": self._contract_attempts,
                "dorisQueries": self._doris_queries,
            },
            "stages": {
                name: {
                    "calls": self._stage_calls.get(name, 0),
                    "totalDurationMs": round(duration * 1000, 3),
                }
                for name, duration in sorted(
                    self._stage_duration_seconds.items()
                )
            },
            "deniedAttempts": [dict(item) for item in self._denied_attempts],
        }


@dataclass
class GroundedSemanticReadLedger:
    """Logical branch-local ownership of immutable semantic documents."""

    evidence_by_ref: dict[str, dict[str, Any]] = field(default_factory=dict)
    ref_by_path: dict[str, str] = field(default_factory=dict)
    lock: Any = field(default_factory=RLock, repr=False)

    def retain(self, evidence: dict[str, Any]) -> bool:
        ref_id = str(evidence.get("refId") or "").strip()
        path = str(evidence.get("path") or "").lstrip("/")
        if not ref_id or not path:
            return False
        with self.lock:
            already_present = ref_id in self.evidence_by_ref
            self.evidence_by_ref[ref_id] = dict(evidence)
            self.ref_by_path[path] = ref_id
        return not already_present

    def has_path(self, path: str) -> bool:
        with self.lock:
            return str(path or "").lstrip("/") in self.ref_by_path

    def evidence(self, ref_ids: list[str] | None = None) -> list[dict[str, Any]]:
        with self.lock:
            if ref_ids is None:
                return [dict(item) for item in self.evidence_by_ref.values()]
            return [
                dict(self.evidence_by_ref[ref_id])
                for ref_id in ref_ids
                if ref_id in self.evidence_by_ref
            ]

    def refs(self) -> list[str]:
        with self.lock:
            return list(self.evidence_by_ref)

    def paths(self) -> list[str]:
        with self.lock:
            return list(self.ref_by_path)


@dataclass
class GroundedQueryBranchContext:
    spec: GroundedQueryBranchSpec
    runtime: Optional[GroundedRuntimeSession]
    budget: GroundedBranchBudget
    semantic_ledger: GroundedSemanticReadLedger = field(
        default_factory=GroundedSemanticReadLedger
    )
    opened_topics: list[str] = field(default_factory=list)
    dependency_query_ids: list[str] = field(default_factory=list)
    dependency_goal_ids: list[str] = field(default_factory=list)
    status: str = "DECLARED"
    last_gaps: list[dict[str, Any]] = field(default_factory=list)
    verified_artifact_ids: list[str] = field(default_factory=list)
    lock: Any = field(default_factory=RLock, repr=False)

    def effective_topics(self) -> list[str]:
        return list(
            dict.fromkeys([*self.spec.topic_scope, *self.opened_topics])
        )

    @property
    def dependent(self) -> bool:
        return bool(self.dependency_query_ids or self.dependency_goal_ids)

    def report(self) -> dict[str, Any]:
        with self.lock:
            return {
                "queryId": self.spec.query_id,
                "objective": self.spec.objective,
                "goalIds": list(self.spec.goal_ids),
                "topicScope": self.effective_topics(),
                "status": self.status,
                "dependencyQueryIds": list(self.dependency_query_ids),
                "dependencyGoalIds": list(self.dependency_goal_ids),
                "semanticRefIds": self.semantic_ledger.refs(),
                "semanticPaths": self.semantic_ledger.paths(),
                "verifiedArtifactIds": list(self.verified_artifact_ids),
                "lastGaps": [dict(item) for item in self.last_gaps],
                "budget": self.budget.report(),
            }
