from __future__ import annotations

import hashlib
import json
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import Field, field_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_query_contract import GroundedBindingHints


class QueryOutcomeStatus(str, Enum):
    VERIFIED = "VERIFIED"
    NEEDS_REASONING = "NEEDS_REASONING"
    DENIED = "DENIED"
    FAILED = "FAILED"


class QueryAttemptStatus(str, Enum):
    VERIFIED = "VERIFIED"
    INTERNAL_REPAIR_REQUIRED = "INTERNAL_REPAIR_REQUIRED"
    NEEDS_REASONING = "NEEDS_REASONING"
    DENIED = "DENIED"
    FAILED = "FAILED"


class QuerySqlCandidate(APIModel):
    sql: str
    rationale: str = ""
    evidence_ref_ids: list[str] = Field(default_factory=list)

    @field_validator("sql")
    @classmethod
    def require_sql(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("query SQL candidate is required")
        return normalized


class QueryRepairReceipt(APIModel):
    version: str = "query_repair_receipt.v1"
    receipt_id: str
    query_id: str
    caller_id: str
    stage: str
    code: str
    attempt_count: int
    contract_generation: int = 0
    contract_fingerprint: str = ""
    sql_ast_fingerprint: str = ""
    allowed_next_actions: list[str] = Field(default_factory=list)
    receipt_fingerprint: str

    def fingerprint_valid(self) -> bool:
        return self.receipt_fingerprint == query_repair_receipt_fingerprint(self)


class QueryRequest(APIModel):
    """Model-authored business query request without tenant authority fields.

    A published ``metric_ref`` is already a complete metric output binding.  A
    caller must not repeat its source column in ``field_aggregations`` or
    ``selected_fields`` unless a separate declared Goal requires that output.
    """

    query_id: str
    goal_ids: list[str] = Field(
        default_factory=list,
        description=(
            "Declared answer Goal IDs proved by this query. TIME_WINDOW Goals "
            "may accompany their data Goal so the Contract can verify scope."
        ),
    )
    read_ref_ids: list[str] = Field(default_factory=list)
    semantic_paths: list[str] = Field(default_factory=list)
    binding_hints: GroundedBindingHints = Field(
        default_factory=GroundedBindingHints,
        description=(
            "Semantic bindings for the query. When metricRefs already cover "
            "all assigned METRIC Goals, do not repeat their source columns in "
            "fieldAggregations or selectedFields."
        ),
    )
    sql_candidate: Optional[QuerySqlCandidate] = None
    reason: str = ""
    repair_receipt: Optional[QueryRepairReceipt] = None

    @field_validator("query_id")
    @classmethod
    def require_query_id(cls, value: str) -> str:
        normalized = str(value or "").strip()
        if not normalized:
            raise ValueError("query_id is required")
        return normalized

    @field_validator("goal_ids", "read_ref_ids", "semantic_paths")
    @classmethod
    def normalize_string_list(cls, value: list[str]) -> list[str]:
        return list(
            dict.fromkeys(
                str(item or "").strip()
                for item in value
                if str(item or "").strip()
            )
        )


class QueryAttemptResult(APIModel):
    status: QueryAttemptStatus
    query_id: str
    stage: str = ""
    code: str = ""
    message: str = ""
    next_actions: list[str] = Field(default_factory=list)
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    read_next: list[dict[str, Any]] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    covered_goal_ids: list[str] = Field(default_factory=list)
    row_count: int = 0
    tables: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    contract_generation: int = 0
    contract_fingerprint: str = ""
    sql_ast_fingerprint: str = ""
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class StructuredQueryObservation(APIModel):
    stage: str
    code: str
    message: str = ""
    retryable: bool = False
    gaps: list[dict[str, Any]] = Field(default_factory=list)
    read_next: list[dict[str, Any]] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    repair_receipt: Optional[QueryRepairReceipt] = None


class CoreQueryObservation(APIModel):
    """Small caller-facing observation; full diagnostics stay in the runtime."""

    stage: str
    code: str
    message: str = ""
    retryable: bool = False
    missing_evidence: list[str] = Field(default_factory=list)
    read_next: list[dict[str, Any]] = Field(default_factory=list)
    allowed_next_actions: list[str] = Field(default_factory=list)
    repair_receipt: Optional[QueryRepairReceipt] = None


class CoreQueryBranchObservation(APIModel):
    """One compact query branch result exposed to Core ReAct."""

    query_id: str
    status: str
    code: str = ""
    message: str = ""
    artifact_ids: list[str] = Field(default_factory=list)
    covered_goal_ids: list[str] = Field(default_factory=list)
    row_count: int = 0
    attempt_count: int = 0
    internal_repair_count: int = 0
    observation: Optional[CoreQueryObservation] = None
    decision_mode: str = "CALLER_REACT"


class CoreQueryBatchObservation(APIModel):
    """Compact batch envelope; artifacts and full receipts remain external."""

    status: str
    summary: str = ""
    decision_mode: str = "CALLER_REACT"
    branches: list[CoreQueryBranchObservation] = Field(default_factory=list)
    # ``outcomes`` is retained as a migration alias for callers that consumed
    # the pre-projection batch envelope. Both lists contain the same compact
    # branch objects; complete QueryOutcome objects never cross the tool edge.
    outcomes: list[CoreQueryBranchObservation] = Field(default_factory=list)
    verified_count: int = 0
    denied_count: int = 0
    needs_reasoning_count: int = 0
    failed_count: int = 0


class QueryOutcome(APIModel):
    status: QueryOutcomeStatus
    query_id: str
    artifact_ids: list[str] = Field(default_factory=list)
    covered_goal_ids: list[str] = Field(default_factory=list)
    row_count: int = 0
    tables: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    internal_repair_count: int = 0
    observation: Optional[StructuredQueryObservation] = None
    diagnostics: dict[str, Any] = Field(default_factory=dict)

    def to_core_observation(self) -> CoreQueryBranchObservation:
        observation = None
        code = ""
        message = ""
        if self.observation is not None:
            source = self.observation
            code = str(source.code or "")[:160]
            message = str(source.message or "")[:320]
            missing_evidence: list[str] = []
            for raw_gap in source.gaps:
                if not isinstance(raw_gap, dict):
                    continue
                for ref_id in raw_gap.get("rejectedRefIds") or []:
                    if str(ref_id or "").strip():
                        missing_evidence.append(str(ref_id).strip())
                required = raw_gap.get("requiredCapability")
                if isinstance(required, dict):
                    ref_id = required.get("semanticRefId") or required.get("refId")
                    if str(ref_id or "").strip():
                        missing_evidence.append(str(ref_id).strip())
            read_next: list[dict[str, Any]] = []
            for raw_item in source.read_next[:8]:
                if not isinstance(raw_item, dict):
                    continue
                item = {
                    key: raw_item[key]
                    for key in (
                        "refId",
                        "path",
                        "kind",
                        "topic",
                        "table",
                    )
                    if raw_item.get(key) not in (None, "", [], {})
                }
                for key in ("refId", "path", "kind", "topic", "table"):
                    if key in item:
                        item[key] = str(item[key])[:240]
                read_next.append(item)
                ref_id = raw_item.get("refId")
                if str(ref_id or "").strip():
                    missing_evidence.append(str(ref_id).strip())
            observation = CoreQueryObservation(
                stage=str(source.stage or "QUERY"),
                code=code or self.status,
                message=message,
                retryable=bool(source.retryable),
                missing_evidence=list(dict.fromkeys(missing_evidence))[:16],
                read_next=read_next,
                allowed_next_actions=list(
                    dict.fromkeys(str(item or "").strip() for item in source.next_actions if str(item or "").strip())
                )[:8],
                repair_receipt=source.repair_receipt,
            )
            # Gap details are retained in the full internal observation.  The
            # Core only needs their compact identifiers and exact read targets.
        return CoreQueryBranchObservation(
            query_id=self.query_id,
            status=self.status.value if isinstance(self.status, Enum) else str(self.status),
            code=code,
            message=message,
            artifact_ids=list(self.artifact_ids),
            covered_goal_ids=list(self.covered_goal_ids),
            row_count=int(self.row_count or 0),
            attempt_count=int(self.attempt_count or 0),
            internal_repair_count=int(self.internal_repair_count or 0),
            observation=observation,
            decision_mode="CALLER_REACT",
        )


class QueryBatchOutcome(APIModel):
    status: str
    outcomes: list[QueryOutcome] = Field(default_factory=list)
    verified_count: int = 0
    denied_count: int = 0
    needs_reasoning_count: int = 0
    failed_count: int = 0

    def to_core_observation(self) -> CoreQueryBatchObservation:
        branches = [item.to_core_observation() for item in self.outcomes]
        summary = "%d branches: %d verified" % (
            len(branches),
            int(self.verified_count or 0),
        )
        if self.needs_reasoning_count:
            summary += ", %d need reasoning" % int(self.needs_reasoning_count)
        if self.failed_count:
            summary += ", %d failed" % int(self.failed_count)
        if self.denied_count:
            summary += ", %d denied" % int(self.denied_count)
        return CoreQueryBatchObservation(
            status=str(self.status or "FAILED"),
            summary=summary,
            decision_mode="CALLER_REACT",
            branches=branches,
            outcomes=branches,
            verified_count=int(self.verified_count or 0),
            denied_count=int(self.denied_count or 0),
            needs_reasoning_count=int(self.needs_reasoning_count or 0),
            failed_count=int(self.failed_count or 0),
        )


def issue_query_repair_receipt(
    *,
    query_id: str,
    caller_id: str,
    stage: str,
    code: str,
    attempt_count: int,
    contract_generation: int = 0,
    contract_fingerprint: str = "",
    sql_ast_fingerprint: str = "",
    allowed_next_actions: list[str] | None = None,
) -> QueryRepairReceipt:
    receipt = QueryRepairReceipt(
        receipt_id="query_repair_%s" % uuid.uuid4().hex[:20],
        query_id=str(query_id or "").strip(),
        caller_id=str(caller_id or "").strip(),
        stage=str(stage or "UNKNOWN").strip().upper(),
        code=str(code or "QUERY_REPAIR_REQUIRED").strip().upper(),
        attempt_count=max(0, int(attempt_count or 0)),
        contract_generation=max(0, int(contract_generation or 0)),
        contract_fingerprint=str(contract_fingerprint or "").strip(),
        sql_ast_fingerprint=str(sql_ast_fingerprint or "").strip(),
        allowed_next_actions=list(
            dict.fromkeys(
                str(item or "").strip()
                for item in (allowed_next_actions or [])
                if str(item or "").strip()
            )
        ),
        receipt_fingerprint="pending",
    )
    return receipt.model_copy(
        update={
            "receipt_fingerprint": query_repair_receipt_fingerprint(receipt)
        }
    )


def query_repair_receipt_fingerprint(receipt: QueryRepairReceipt) -> str:
    payload = receipt.model_dump(
        by_alias=True,
        mode="json",
        exclude={"receipt_fingerprint"},
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
