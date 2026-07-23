from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from merchant_ai.services.query_request import (
    QueryAttemptResult,
    QueryAttemptStatus,
    QueryBatchOutcome,
    QueryOutcome,
    QueryOutcomeStatus,
    QueryRequest,
    QuerySqlCandidate,
    StructuredQueryObservation,
    issue_query_repair_receipt,
)


class GovernedQueryBackend(Protocol):
    """Runtime adapter around the existing governed query stages."""

    def execute_attempt(
        self,
        request: QueryRequest,
        *,
        attempt_number: int,
    ) -> QueryAttemptResult: ...

    def repair_request(
        self,
        request: QueryRequest,
        result: QueryAttemptResult,
        *,
        repair_number: int,
    ) -> Optional[QueryRequest]: ...


class GovernedQueryService:
    """One governed query facade with bounded local repair.

    Tenant, ACL, SQL validation, execution and evidence authority stay in the
    injected backend. This service owns only the stable request/outcome
    protocol and the bounded repair loop.
    """

    def __init__(self, *, max_internal_repairs: int = 2, max_batch_workers: int = 4):
        self.max_internal_repairs = max(0, min(int(max_internal_repairs or 0), 4))
        self.max_batch_workers = max(1, min(int(max_batch_workers or 1), 8))

    def execute(
        self,
        request: QueryRequest,
        *,
        backend: GovernedQueryBackend,
        caller_id: str,
    ) -> QueryOutcome:
        normalized = (
            request
            if isinstance(request, QueryRequest)
            else QueryRequest.model_validate(request)
        )
        receipt_error = self._repair_receipt_error(normalized, caller_id)
        if receipt_error:
            return self._denied_outcome(
                normalized,
                caller_id=caller_id,
                code=receipt_error,
            )

        current = normalized
        repairs = 0
        attempts = 0
        while True:
            attempts += 1
            result = backend.execute_attempt(
                current,
                attempt_number=attempts,
            )
            if not isinstance(result, QueryAttemptResult):
                result = QueryAttemptResult.model_validate(result)
            if result.query_id != normalized.query_id:
                return self._failed_outcome(
                    normalized,
                    caller_id=caller_id,
                    code="QUERY_BACKEND_IDENTITY_MISMATCH",
                    message="query backend returned a different query_id",
                    attempts=attempts,
                    repairs=repairs,
                )
            if result.status == QueryAttemptStatus.VERIFIED:
                return QueryOutcome(
                    status=QueryOutcomeStatus.VERIFIED,
                    query_id=normalized.query_id,
                    artifact_ids=result.artifact_ids,
                    covered_goal_ids=result.covered_goal_ids,
                    row_count=result.row_count,
                    tables=result.tables,
                    output_columns=result.output_columns,
                    attempt_count=attempts,
                    internal_repair_count=repairs,
                    diagnostics=result.diagnostics,
                )
            if result.status == QueryAttemptStatus.DENIED:
                return self._outcome_from_result(
                    normalized,
                    result,
                    caller_id=caller_id,
                    status=QueryOutcomeStatus.DENIED,
                    attempts=attempts,
                    repairs=repairs,
                    retryable=False,
                )
            if result.status == QueryAttemptStatus.FAILED:
                return self._outcome_from_result(
                    normalized,
                    result,
                    caller_id=caller_id,
                    status=QueryOutcomeStatus.FAILED,
                    attempts=attempts,
                    repairs=repairs,
                    retryable=False,
                )
            if result.status == QueryAttemptStatus.NEEDS_REASONING:
                return self._outcome_from_result(
                    normalized,
                    result,
                    caller_id=caller_id,
                    status=QueryOutcomeStatus.NEEDS_REASONING,
                    attempts=attempts,
                    repairs=repairs,
                    retryable=True,
                )

            if repairs >= self.max_internal_repairs:
                exhausted = result.model_copy(
                    update={
                        "status": QueryAttemptStatus.NEEDS_REASONING,
                        "code": "QUERY_INTERNAL_REPAIR_EXHAUSTED",
                        "message": (
                            result.message
                            or "query_data exhausted its bounded internal repair budget"
                        ),
                        "next_actions": list(
                            dict.fromkeys(
                                [
                                    *result.next_actions,
                                    "REVISE_QUERY_REQUEST",
                                ]
                            )
                        ),
                    }
                )
                return self._outcome_from_result(
                    normalized,
                    exhausted,
                    caller_id=caller_id,
                    status=QueryOutcomeStatus.NEEDS_REASONING,
                    attempts=attempts,
                    repairs=repairs,
                    retryable=True,
                )

            repaired = backend.repair_request(
                current,
                result,
                repair_number=repairs + 1,
            )
            if repaired is None:
                needs_reasoning = result.model_copy(
                    update={
                        "status": QueryAttemptStatus.NEEDS_REASONING,
                        "next_actions": list(
                            dict.fromkeys(
                                [
                                    *result.next_actions,
                                    "REVISE_QUERY_REQUEST",
                                ]
                            )
                        ),
                    }
                )
                return self._outcome_from_result(
                    normalized,
                    needs_reasoning,
                    caller_id=caller_id,
                    status=QueryOutcomeStatus.NEEDS_REASONING,
                    attempts=attempts,
                    repairs=repairs,
                    retryable=True,
                )
            current = (
                repaired
                if isinstance(repaired, QueryRequest)
                else QueryRequest.model_validate(repaired)
            )
            if current.query_id != normalized.query_id:
                return self._failed_outcome(
                    normalized,
                    caller_id=caller_id,
                    code="QUERY_REPAIR_IDENTITY_MISMATCH",
                    message="internal query repair changed query_id",
                    attempts=attempts,
                    repairs=repairs,
                )
            repairs += 1

    def execute_batch(
        self,
        requests: Sequence[QueryRequest],
        *,
        backend_factory: Protocol,
        caller_id: str,
    ) -> QueryBatchOutcome:
        normalized = [
            item if isinstance(item, QueryRequest) else QueryRequest.model_validate(item)
            for item in requests
        ]
        if not normalized:
            return QueryBatchOutcome(status="REJECTED")
        query_ids = [item.query_id for item in normalized]
        if len(set(query_ids)) != len(query_ids):
            return QueryBatchOutcome(
                status="REJECTED",
                outcomes=[
                    self._failed_outcome(
                        item,
                        caller_id=caller_id,
                        code="QUERY_BATCH_ID_DUPLICATE",
                        message="query_batch requires unique query_id values",
                        attempts=0,
                        repairs=0,
                    )
                    for item in normalized
                ],
                failed_count=len(normalized),
            )

        outcomes: list[QueryOutcome] = []
        with ThreadPoolExecutor(
            max_workers=min(len(normalized), self.max_batch_workers),
            thread_name_prefix="governed-query",
        ) as pool:
            futures = {
                pool.submit(
                    self.execute,
                    item,
                    backend=backend_factory(item),
                    caller_id=caller_id,
                ): item
                for item in normalized
            }
            by_id: dict[str, QueryOutcome] = {}
            for future in as_completed(futures):
                request = futures[future]
                try:
                    by_id[request.query_id] = future.result()
                except Exception as exc:
                    by_id[request.query_id] = self._failed_outcome(
                        request,
                        caller_id=caller_id,
                        code="QUERY_BATCH_BRANCH_INTERNAL_ERROR",
                        message="%s:%s"
                        % (type(exc).__name__, str(exc)[:500]),
                        attempts=1,
                        repairs=0,
                    )
        outcomes = [by_id[query_id] for query_id in query_ids]
        verified = sum(item.status == QueryOutcomeStatus.VERIFIED for item in outcomes)
        denied = sum(item.status == QueryOutcomeStatus.DENIED for item in outcomes)
        needs_reasoning = sum(
            item.status == QueryOutcomeStatus.NEEDS_REASONING for item in outcomes
        )
        failed = sum(item.status == QueryOutcomeStatus.FAILED for item in outcomes)
        return QueryBatchOutcome(
            status=(
                "VERIFIED"
                if verified == len(outcomes)
                else "PARTIAL"
                if verified
                else "DENIED"
                if denied == len(outcomes)
                else "NEEDS_REASONING"
                if needs_reasoning
                else "FAILED"
            ),
            outcomes=outcomes,
            verified_count=verified,
            denied_count=denied,
            needs_reasoning_count=needs_reasoning,
            failed_count=failed,
        )

    def outcome_from_attempt(
        self,
        request: QueryRequest,
        result: QueryAttemptResult,
        *,
        caller_id: str,
    ) -> QueryOutcome:
        """Normalize an already executed batch branch into the public protocol."""

        if result.status == QueryAttemptStatus.VERIFIED:
            return QueryOutcome(
                status=QueryOutcomeStatus.VERIFIED,
                query_id=request.query_id,
                artifact_ids=result.artifact_ids,
                covered_goal_ids=result.covered_goal_ids,
                row_count=result.row_count,
                tables=result.tables,
                output_columns=result.output_columns,
                attempt_count=1,
                diagnostics=result.diagnostics,
            )
        public_status = (
            QueryOutcomeStatus.DENIED
            if result.status == QueryAttemptStatus.DENIED
            else QueryOutcomeStatus.FAILED
            if result.status == QueryAttemptStatus.FAILED
            else QueryOutcomeStatus.NEEDS_REASONING
        )
        normalized = result
        if result.status == QueryAttemptStatus.INTERNAL_REPAIR_REQUIRED:
            normalized = result.model_copy(
                update={
                    "status": QueryAttemptStatus.NEEDS_REASONING,
                    "code": result.code or "QUERY_BATCH_REPAIR_REQUIRED",
                    "next_actions": list(
                        dict.fromkeys(
                            [*result.next_actions, "RETRY_QUERY_DATA_SERIAL"]
                        )
                    ),
                }
            )
        return self._outcome_from_result(
            request,
            normalized,
            caller_id=caller_id,
            status=public_status,
            attempts=1,
            repairs=0,
            retryable=public_status == QueryOutcomeStatus.NEEDS_REASONING,
        )

    @staticmethod
    def _repair_receipt_error(request: QueryRequest, caller_id: str) -> str:
        receipt = request.repair_receipt
        if receipt is None:
            return ""
        if not receipt.fingerprint_valid():
            return "QUERY_REPAIR_RECEIPT_INVALID"
        if receipt.query_id != request.query_id:
            return "QUERY_REPAIR_RECEIPT_QUERY_MISMATCH"
        if receipt.caller_id != str(caller_id or "").strip():
            return "QUERY_REPAIR_RECEIPT_CALLER_MISMATCH"
        return ""

    @staticmethod
    def _outcome_from_result(
        request: QueryRequest,
        result: QueryAttemptResult,
        *,
        caller_id: str,
        status: QueryOutcomeStatus,
        attempts: int,
        repairs: int,
        retryable: bool,
    ) -> QueryOutcome:
        receipt = None
        if status == QueryOutcomeStatus.NEEDS_REASONING:
            receipt = issue_query_repair_receipt(
                query_id=request.query_id,
                caller_id=caller_id,
                stage=result.stage,
                code=result.code,
                attempt_count=attempts,
                contract_generation=result.contract_generation,
                contract_fingerprint=result.contract_fingerprint,
                sql_ast_fingerprint=result.sql_ast_fingerprint,
                allowed_next_actions=result.next_actions,
            )
        return QueryOutcome(
            status=status,
            query_id=request.query_id,
            artifact_ids=result.artifact_ids,
            covered_goal_ids=result.covered_goal_ids,
            row_count=result.row_count,
            tables=result.tables,
            output_columns=result.output_columns,
            attempt_count=attempts,
            internal_repair_count=repairs,
            observation=StructuredQueryObservation(
                stage=result.stage or "QUERY",
                code=result.code or status.value,
                message=result.message,
                retryable=retryable,
                gaps=result.gaps,
                read_next=result.read_next,
                next_actions=result.next_actions,
                repair_receipt=receipt,
            ),
            diagnostics=result.diagnostics,
        )

    def _denied_outcome(
        self,
        request: QueryRequest,
        *,
        caller_id: str,
        code: str,
    ) -> QueryOutcome:
        return self._outcome_from_result(
            request,
            QueryAttemptResult(
                status=QueryAttemptStatus.DENIED,
                query_id=request.query_id,
                stage="REQUEST",
                code=code,
                message="query repair authority is invalid for this caller",
            ),
            caller_id=caller_id,
            status=QueryOutcomeStatus.DENIED,
            attempts=0,
            repairs=0,
            retryable=False,
        )

    def _failed_outcome(
        self,
        request: QueryRequest,
        *,
        caller_id: str,
        code: str,
        message: str,
        attempts: int,
        repairs: int,
    ) -> QueryOutcome:
        return self._outcome_from_result(
            request,
            QueryAttemptResult(
                status=QueryAttemptStatus.FAILED,
                query_id=request.query_id,
                stage="INTERNAL",
                code=code,
                message=message,
            ),
            caller_id=caller_id,
            status=QueryOutcomeStatus.FAILED,
            attempts=attempts,
            repairs=repairs,
            retryable=False,
        )


class CallbackGroundedQueryBackend:
    """Internal adapter that collapses governed query stages behind callbacks.

    The callbacks are runtime services rather than caller-visible tools. This
    keeps ``query_data`` as the public facade while preserving independently
    testable deterministic stages.
    """

    def __init__(
        self,
        *,
        prepare_contract: Callable[[QueryRequest], Mapping[str, Any]],
        execute_compiled: Callable[[QueryRequest], Mapping[str, Any]],
        submit_sql: Callable[
            [QueryRequest, QuerySqlCandidate, Mapping[str, Any]],
            Mapping[str, Any],
        ],
        repair_sql: Optional[
            Callable[
                [QueryRequest, QueryAttemptResult, int],
                Optional[QuerySqlCandidate],
            ]
        ] = None,
    ) -> None:
        self.prepare_contract = prepare_contract
        self.execute_compiled = execute_compiled
        self.submit_sql = submit_sql
        self.repair_sql = repair_sql
        self.prepared: dict[str, Any] = {}

    def execute_attempt(
        self,
        request: QueryRequest,
        *,
        attempt_number: int,
    ) -> QueryAttemptResult:
        if not self.prepared:
            prepared = dict(self.prepare_contract(request))
            prepared_result = self._prepared_result(request, prepared)
            if prepared_result is not None:
                return prepared_result
            self.prepared = prepared

        execution_mode = str(
            self.prepared.get("executionMode")
            or self.prepared.get("execution_mode")
            or ""
        ).upper()
        if execution_mode == "CORE_SQL_REQUIRED":
            candidate = request.sql_candidate
            if candidate is None:
                return QueryAttemptResult(
                    status=QueryAttemptStatus.NEEDS_REASONING,
                    query_id=request.query_id,
                    stage="SQL_GENERATION",
                    code="CORE_SQL_CANDIDATE_REQUIRED",
                    message="the active complex Contract requires a complete SQL candidate",
                    next_actions=["AUTHOR_SQL_AND_RETRY_QUERY_DATA"],
                    contract_generation=int(
                        self.prepared.get("activeGeneration") or 0
                    ),
                    contract_fingerprint=str(
                        self.prepared.get("contractFingerprint") or ""
                    ),
                    diagnostics={
                        "executionMode": execution_mode,
                        "sqlObligations": self.prepared.get("sqlObligations") or {},
                    },
                )
            payload = dict(self.submit_sql(request, candidate, self.prepared))
            return self._terminal_result(
                request,
                payload,
                stage="SQL_VALIDATION",
            )

        payload = dict(self.execute_compiled(request))
        return self._terminal_result(request, payload, stage="EXECUTION")

    def repair_request(
        self,
        request: QueryRequest,
        result: QueryAttemptResult,
        *,
        repair_number: int,
    ) -> Optional[QueryRequest]:
        if self.repair_sql is None:
            return None
        candidate = self.repair_sql(request, result, repair_number)
        if candidate is None:
            return None
        if not isinstance(candidate, QuerySqlCandidate):
            candidate = QuerySqlCandidate.model_validate(candidate)
        return request.model_copy(update={"sql_candidate": candidate})

    @staticmethod
    def _prepared_result(
        request: QueryRequest,
        payload: Mapping[str, Any],
    ) -> Optional[QueryAttemptResult]:
        if _legacy_payload_denied(payload):
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.DENIED,
                stage="CONTRACT",
            )
        activated = bool(payload.get("activated"))
        status = str(payload.get("status") or "").upper()
        if activated and status in {"READY", "PREPARED"}:
            return None
        if activated and not status:
            return None
        if _legacy_payload_internal_failure(payload):
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.FAILED,
                stage="CONTRACT",
            )
        return _legacy_query_attempt(
            request,
            payload,
            status=QueryAttemptStatus.NEEDS_REASONING,
            stage="CONTRACT",
        )

    @staticmethod
    def _terminal_result(
        request: QueryRequest,
        payload: Mapping[str, Any],
        *,
        stage: str,
    ) -> QueryAttemptResult:
        status = str(payload.get("status") or "").upper()
        if status == "VERIFIED":
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.VERIFIED,
                stage="EVIDENCE",
            )
        if _legacy_payload_denied(payload):
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.DENIED,
                stage=stage,
            )
        next_action = str(payload.get("nextAction") or "").upper()
        repair_statuses = {
            "REJECTED",
            "NO_PROGRESS",
            "SQL_EXECUTION_REPAIR_REQUIRED",
        }
        if status in repair_statuses and next_action in {
            "REPAIR_SQL",
            "SUBMIT_GROUNDED_SQL_CANDIDATE",
        }:
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.INTERNAL_REPAIR_REQUIRED,
                stage="SQL_VALIDATION" if status != "SQL_EXECUTION_REPAIR_REQUIRED" else "EXECUTION",
            )
        if _legacy_payload_internal_failure(payload):
            return _legacy_query_attempt(
                request,
                payload,
                status=QueryAttemptStatus.FAILED,
                stage=stage,
            )
        return _legacy_query_attempt(
            request,
            payload,
            status=QueryAttemptStatus.NEEDS_REASONING,
            stage=stage,
        )


def classify_query_stage_payload(
    request: QueryRequest,
    payload: Mapping[str, Any],
    *,
    stage: str,
) -> QueryAttemptResult:
    if str(stage or "").upper() == "CONTRACT":
        result = CallbackGroundedQueryBackend._prepared_result(request, payload)
        if result is not None:
            return result
        return QueryAttemptResult(
            status=QueryAttemptStatus.NEEDS_REASONING,
            query_id=request.query_id,
            stage="CONTRACT",
            code="QUERY_BATCH_PREPARED_NOT_EXECUTED",
            next_actions=["EXECUTE_QUERY_BATCH"],
        )
    return CallbackGroundedQueryBackend._terminal_result(
        request,
        payload,
        stage=stage,
    )


def _legacy_query_attempt(
    request: QueryRequest,
    payload: Mapping[str, Any],
    *,
    status: QueryAttemptStatus,
    stage: str,
) -> QueryAttemptResult:
    artifact_ids = [
        str(item or "").strip()
        for item in [
            payload.get("queryArtifactId"),
            *(payload.get("adoptedArtifactIds") or []),
        ]
        if str(item or "").strip()
    ]
    raw_next = payload.get("nextActions") or payload.get("nextAction") or []
    if isinstance(raw_next, str):
        raw_next = [raw_next]
    raw_gaps = payload.get("gaps") or payload.get("blockingGaps") or []
    if not isinstance(raw_gaps, list):
        raw_gaps = [raw_gaps] if isinstance(raw_gaps, dict) else []
    raw_read_next = payload.get("readNext") or []
    if not isinstance(raw_read_next, list):
        raw_read_next = [raw_read_next] if isinstance(raw_read_next, dict) else []
    code = str(payload.get("code") or "").strip().upper()
    if not code and status != QueryAttemptStatus.VERIFIED:
        code = str(payload.get("status") or status.value).strip().upper()
    return QueryAttemptResult(
        status=status,
        query_id=request.query_id,
        stage=stage,
        code=code,
        message=str(payload.get("message") or payload.get("instruction") or "")[:1000],
        next_actions=[
            str(item or "").strip()
            for item in raw_next
            if str(item or "").strip()
        ],
        gaps=[dict(item) for item in raw_gaps if isinstance(item, Mapping)],
        read_next=[
            dict(item) for item in raw_read_next if isinstance(item, Mapping)
        ],
        artifact_ids=list(dict.fromkeys(artifact_ids)),
        covered_goal_ids=[
            str(item or "").strip()
            for item in payload.get("coveredGoalIds") or []
            if str(item or "").strip()
        ],
        row_count=max(0, int(payload.get("rowCount") or 0)),
        tables=[
            str(item or "").strip()
            for item in payload.get("tables") or []
            if str(item or "").strip()
        ],
        output_columns=[
            str(item or "").strip()
            for item in payload.get("outputColumns") or []
            if str(item or "").strip()
        ],
        contract_generation=max(
            0,
            int(
                payload.get("activeGeneration")
                or payload.get("contractGeneration")
                or 0
            ),
        ),
        contract_fingerprint=str(payload.get("contractFingerprint") or ""),
        sql_ast_fingerprint=str(payload.get("astFingerprint") or ""),
        diagnostics={
            key: payload.get(key)
            for key in (
                "executionMode",
                "sqlObligations",
                "repairReview",
                "replanEvidence",
                "failureDisposition",
                "resultArtifacts",
            )
            if payload.get(key) not in (None, "", [], {})
        },
    )


def _legacy_payload_denied(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status") or "").upper()
    code = str(payload.get("code") or "").upper()
    disposition = str(payload.get("failureDisposition") or "").upper()
    return (
        status in {"ACCESS_DENIED", "DENIED"}
        or disposition == "SECURITY_TERMINAL"
        or code
        in {
            "ACCESS_DENIED",
            "ACL_POLICY_UNAVAILABLE",
            "ACL_SQL_PARSE_FAILED",
            "MERCHANT_SCOPE_DENIED",
            "TABLE_DENIED",
            "TABLE_NOT_ALLOWED",
            "TABLE_ROLE_DENIED",
            "COLUMN_DENIED",
        }
    )


def _legacy_payload_internal_failure(payload: Mapping[str, Any]) -> bool:
    status = str(payload.get("status") or "").upper()
    code = str(payload.get("code") or "").upper()
    next_action = str(payload.get("nextAction") or "").upper()
    return (
        next_action in {"STOP", "STOP_INTERNAL"}
        or status in {"OPERATIONAL_FAILURE", "FAILED"}
        or "INTERNAL_ERROR" in code
        or "VALIDATOR_INTERNAL_ERROR" in code
    )
