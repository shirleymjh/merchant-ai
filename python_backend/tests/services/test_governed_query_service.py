from __future__ import annotations

from dataclasses import dataclass, field

from merchant_ai.services.governed_query_service import (
    CallbackGroundedQueryBackend,
    GovernedQueryService,
)
from merchant_ai.services.query_request import (
    QueryAttemptResult,
    QueryAttemptStatus,
    QueryBatchOutcome,
    QueryOutcome,
    QueryOutcomeStatus,
    QueryRequest,
    QuerySqlCandidate,
    StructuredQueryObservation,
)


@dataclass
class ScriptedBackend:
    results: list[QueryAttemptResult]
    repaired_sql: str = "SELECT 1"
    calls: int = 0
    repairs: int = 0
    requests: list[QueryRequest] = field(default_factory=list)

    def execute_attempt(
        self,
        request: QueryRequest,
        *,
        attempt_number: int,
    ) -> QueryAttemptResult:
        self.calls += 1
        self.requests.append(request)
        return self.results[min(attempt_number - 1, len(self.results) - 1)]

    def repair_request(
        self,
        request: QueryRequest,
        result: QueryAttemptResult,
        *,
        repair_number: int,
    ) -> QueryRequest:
        self.repairs += 1
        return request.model_copy(
            update={
                "reason": "internal repair %d" % repair_number,
            }
        )


def attempt(
    query_id: str,
    status: QueryAttemptStatus,
    *,
    code: str = "",
) -> QueryAttemptResult:
    return QueryAttemptResult(
        status=status,
        query_id=query_id,
        stage="SQL_VALIDATION",
        code=code,
        artifact_ids=["artifact_%s" % query_id]
        if status == QueryAttemptStatus.VERIFIED
        else [],
        covered_goal_ids=["goal_%s" % query_id]
        if status == QueryAttemptStatus.VERIFIED
        else [],
        row_count=1 if status == QueryAttemptStatus.VERIFIED else 0,
    )


def test_execute_returns_verified_artifact_without_repair() -> None:
    backend = ScriptedBackend([attempt("q1", QueryAttemptStatus.VERIFIED)])

    outcome = GovernedQueryService().execute(
        QueryRequest(query_id="q1", goal_ids=["goal_q1"]),
        backend=backend,
        caller_id="core",
    )

    assert outcome.status == QueryOutcomeStatus.VERIFIED
    assert outcome.artifact_ids == ["artifact_q1"]
    assert outcome.attempt_count == 1
    assert outcome.internal_repair_count == 0
    assert backend.calls == 1


def test_execute_repairs_locally_before_returning_verified() -> None:
    backend = ScriptedBackend(
        [
            attempt(
                "q1",
                QueryAttemptStatus.INTERNAL_REPAIR_REQUIRED,
                code="SQL_ALIAS_INVALID",
            ),
            attempt("q1", QueryAttemptStatus.VERIFIED),
        ]
    )

    outcome = GovernedQueryService(max_internal_repairs=2).execute(
        QueryRequest(query_id="q1"),
        backend=backend,
        caller_id="subagent:analysis-1",
    )

    assert outcome.status == QueryOutcomeStatus.VERIFIED
    assert outcome.attempt_count == 2
    assert outcome.internal_repair_count == 1
    assert backend.repairs == 1


def test_execute_returns_bound_repair_receipt_after_budget_exhaustion() -> None:
    backend = ScriptedBackend(
        [
            attempt(
                "q1",
                QueryAttemptStatus.INTERNAL_REPAIR_REQUIRED,
                code="SQL_VALIDATION_FAILED",
            )
        ]
    )

    outcome = GovernedQueryService(max_internal_repairs=2).execute(
        QueryRequest(query_id="q1"),
        backend=backend,
        caller_id="subagent:analysis-1",
    )

    assert outcome.status == QueryOutcomeStatus.NEEDS_REASONING
    assert outcome.internal_repair_count == 2
    assert outcome.attempt_count == 3
    assert outcome.observation is not None
    assert outcome.observation.code == "QUERY_INTERNAL_REPAIR_EXHAUSTED"
    receipt = outcome.observation.repair_receipt
    assert receipt is not None
    assert receipt.query_id == "q1"
    assert receipt.caller_id == "subagent:analysis-1"
    assert receipt.fingerprint_valid()


def test_semantic_gap_returns_to_caller_without_internal_repair() -> None:
    backend = ScriptedBackend(
        [
            QueryAttemptResult(
                status=QueryAttemptStatus.NEEDS_REASONING,
                query_id="q1",
                stage="CONTRACT",
                code="METRIC_REF_NOT_READ",
                read_next=[{"refId": "metric.refund_rate"}],
                next_actions=["READ_SEMANTIC_ASSET"],
            )
        ]
    )

    outcome = GovernedQueryService().execute(
        QueryRequest(query_id="q1"),
        backend=backend,
        caller_id="core",
    )

    assert outcome.status == QueryOutcomeStatus.NEEDS_REASONING
    assert outcome.internal_repair_count == 0
    assert backend.repairs == 0
    assert outcome.observation is not None
    assert outcome.observation.read_next == [{"refId": "metric.refund_rate"}]


def test_repair_receipt_cannot_move_between_callers() -> None:
    first_backend = ScriptedBackend(
        [
            attempt(
                "q1",
                QueryAttemptStatus.NEEDS_REASONING,
                code="SEMANTIC_GAP",
            )
        ]
    )
    service = GovernedQueryService()
    first = service.execute(
        QueryRequest(query_id="q1"),
        backend=first_backend,
        caller_id="subagent:a",
    )
    assert first.observation is not None
    receipt = first.observation.repair_receipt
    assert receipt is not None

    second_backend = ScriptedBackend(
        [attempt("q1", QueryAttemptStatus.VERIFIED)]
    )
    second = service.execute(
        QueryRequest(query_id="q1", repair_receipt=receipt),
        backend=second_backend,
        caller_id="subagent:b",
    )

    assert second.status == QueryOutcomeStatus.DENIED
    assert second_backend.calls == 0
    assert second.observation is not None
    assert second.observation.code == "QUERY_REPAIR_RECEIPT_CALLER_MISMATCH"


def test_batch_preserves_request_order_and_summarizes_statuses() -> None:
    scripted = {
        "q1": ScriptedBackend([attempt("q1", QueryAttemptStatus.VERIFIED)]),
        "q2": ScriptedBackend(
            [attempt("q2", QueryAttemptStatus.DENIED, code="ACCESS_DENIED")]
        ),
    }

    outcome = GovernedQueryService(max_batch_workers=2).execute_batch(
        [QueryRequest(query_id="q1"), QueryRequest(query_id="q2")],
        backend_factory=lambda request: scripted[request.query_id],
        caller_id="core",
    )

    assert outcome.status == "PARTIAL"
    assert [item.query_id for item in outcome.outcomes] == ["q1", "q2"]
    assert outcome.verified_count == 1
    assert outcome.denied_count == 1


def test_batch_repairs_only_the_failed_branch_and_preserves_success() -> None:
    scripted = {
        "stable": ScriptedBackend(
            [attempt("stable", QueryAttemptStatus.VERIFIED)]
        ),
        "repairable": ScriptedBackend(
            [
                attempt(
                    "repairable",
                    QueryAttemptStatus.INTERNAL_REPAIR_REQUIRED,
                    code="SQL_ALIAS_INVALID",
                ),
                attempt("repairable", QueryAttemptStatus.VERIFIED),
            ]
        ),
        "semantic_gap": ScriptedBackend(
            [
                attempt(
                    "semantic_gap",
                    QueryAttemptStatus.NEEDS_REASONING,
                    code="METRIC_REF_NOT_READ",
                )
            ]
        ),
    }

    outcome = GovernedQueryService(
        max_internal_repairs=2,
        max_batch_workers=3,
    ).execute_batch(
        [QueryRequest(query_id=query_id) for query_id in scripted],
        backend_factory=lambda request: scripted[request.query_id],
        caller_id="core",
    )

    assert outcome.status == "PARTIAL"
    assert outcome.verified_count == 2
    assert outcome.needs_reasoning_count == 1
    assert scripted["stable"].calls == 1
    assert scripted["stable"].repairs == 0
    assert scripted["repairable"].calls == 2
    assert scripted["repairable"].repairs == 1
    assert scripted["semantic_gap"].calls == 1
    assert scripted["semantic_gap"].repairs == 0


def test_batch_contains_one_branch_backend_exception() -> None:
    class ExplodingBackend(ScriptedBackend):
        def execute_attempt(
            self,
            request: QueryRequest,
            *,
            attempt_number: int,
        ) -> QueryAttemptResult:
            raise RuntimeError("branch unavailable")

    scripted = {
        "stable": ScriptedBackend(
            [attempt("stable", QueryAttemptStatus.VERIFIED)]
        ),
        "exploding": ExplodingBackend([]),
    }

    outcome = GovernedQueryService(max_batch_workers=2).execute_batch(
        [QueryRequest(query_id=query_id) for query_id in scripted],
        backend_factory=lambda request: scripted[request.query_id],
        caller_id="core",
    )

    assert outcome.status == "PARTIAL"
    assert outcome.verified_count == 1
    assert outcome.failed_count == 1
    failed = outcome.outcomes[1]
    assert failed.status == QueryOutcomeStatus.FAILED
    assert failed.observation is not None
    assert failed.observation.code == "QUERY_BATCH_BRANCH_INTERNAL_ERROR"


def test_legacy_backend_collapses_prepare_and_deterministic_execution() -> None:
    calls: list[str] = []
    backend = CallbackGroundedQueryBackend(
        prepare_contract=lambda request: {
            "status": "READY",
            "activated": True,
            "executionMode": "DETERMINISTIC",
            "activeGeneration": 1,
            "contractFingerprint": "contract-1",
        },
        execute_compiled=lambda request: calls.append("execute")
        or {
            "status": "VERIFIED",
            "queryArtifactId": "artifact-1",
            "coveredGoalIds": ["goal-1"],
            "rowCount": 3,
        },
        submit_sql=lambda request, candidate, prepared: {},
    )

    outcome = GovernedQueryService().execute(
        QueryRequest(query_id="q1", goal_ids=["goal-1"]),
        backend=backend,
        caller_id="core",
    )

    assert outcome.status == QueryOutcomeStatus.VERIFIED
    assert outcome.artifact_ids == ["artifact-1"]
    assert calls == ["execute"]


def test_legacy_backend_requests_sql_for_complex_contract() -> None:
    backend = CallbackGroundedQueryBackend(
        prepare_contract=lambda request: {
            "status": "READY",
            "activated": True,
            "executionMode": "CORE_SQL_REQUIRED",
            "activeGeneration": 2,
            "contractFingerprint": "contract-2",
            "sqlObligations": {"requiredTables": ["orders"]},
        },
        execute_compiled=lambda request: {},
        submit_sql=lambda request, candidate, prepared: {},
    )

    outcome = GovernedQueryService().execute(
        QueryRequest(query_id="q1"),
        backend=backend,
        caller_id="core",
    )

    assert outcome.status == QueryOutcomeStatus.NEEDS_REASONING
    assert outcome.observation is not None
    assert outcome.observation.code == "CORE_SQL_CANDIDATE_REQUIRED"
    assert outcome.diagnostics["sqlObligations"] == {
        "requiredTables": ["orders"]
    }


def test_legacy_backend_repairs_rejected_sql_inside_query_data() -> None:
    submitted: list[str] = []

    def submit_sql(request, candidate, prepared):
        submitted.append(candidate.sql)
        if len(submitted) == 1:
            return {
                "status": "REJECTED",
                "nextAction": "REPAIR_SQL",
                "gaps": [{"code": "SQL_ALIAS_INVALID"}],
                "activeGeneration": 2,
                "contractFingerprint": "contract-2",
            }
        return {
            "status": "VERIFIED",
            "queryArtifactId": "artifact-2",
            "coveredGoalIds": ["goal-2"],
        }

    backend = CallbackGroundedQueryBackend(
        prepare_contract=lambda request: {
            "status": "READY",
            "activated": True,
            "executionMode": "CORE_SQL_REQUIRED",
            "activeGeneration": 2,
            "contractFingerprint": "contract-2",
        },
        execute_compiled=lambda request: {},
        submit_sql=submit_sql,
        repair_sql=lambda request, result, repair_number: QuerySqlCandidate(
            sql="SELECT 2",
            rationale="repair alias",
        ),
    )

    outcome = GovernedQueryService().execute(
        QueryRequest(
            query_id="q2",
            goal_ids=["goal-2"],
            sql_candidate=QuerySqlCandidate(sql="SELECT 1"),
        ),
        backend=backend,
        caller_id="core",
    )

    assert outcome.status == QueryOutcomeStatus.VERIFIED
    assert outcome.internal_repair_count == 1
    assert submitted == ["SELECT 1", "SELECT 2"]


def test_core_observation_drops_full_diagnostics_and_gap_payload() -> None:
    outcome = QueryOutcome(
        status=QueryOutcomeStatus.NEEDS_REASONING,
        query_id="q-gap",
        attempt_count=2,
        internal_repair_count=1,
        diagnostics={
            "sql": "SELECT a very long internal statement",
            "trace": {"all": "internal-only"},
        },
        observation=StructuredQueryObservation(
            stage="CONTRACT",
            code="METRIC_REF_NOT_READ",
            message="read the exact semantic definition before retrying",
            retryable=True,
            gaps=[
                {
                    "code": "METRIC_REF_NOT_READ",
                    "message": "full internal gap explanation",
                    "rejectedRefIds": ["semantic:orders:metric:gmv"],
                    "requiredCapability": {
                        "semanticRefId": "semantic:orders:metric:gmv"
                    },
                    "internalTrace": {"shouldNotCrossBoundary": True},
                }
            ],
            read_next=[
                {
                    "refId": "semantic:orders:metric:gmv",
                    "path": "/knowledge/orders/metric/gmv",
                    "kind": "METRIC",
                    "internalPayload": "should not cross boundary",
                }
            ],
            next_actions=["READ_SEMANTIC_ASSET", "RETRY_QUERY_DATA"],
        ),
    )

    payload = outcome.to_core_observation().model_dump(
        by_alias=True,
        mode="json",
    )

    assert payload["status"] == "NEEDS_REASONING"
    assert payload["attemptCount"] == 2
    assert payload["internalRepairCount"] == 1
    assert "diagnostics" not in payload
    assert "gaps" not in payload["observation"]
    assert payload["observation"]["missingEvidence"] == [
        "semantic:orders:metric:gmv"
    ]
    assert payload["observation"]["readNext"] == [
        {
            "refId": "semantic:orders:metric:gmv",
            "path": "/knowledge/orders/metric/gmv",
            "kind": "METRIC",
        }
    ]


def test_core_batch_observation_keeps_independent_compact_branches() -> None:
    outcome = QueryBatchOutcome(
        status="PARTIAL",
        outcomes=[
            QueryOutcome(
                status=QueryOutcomeStatus.VERIFIED,
                query_id="orders",
                artifact_ids=["artifact-orders"],
                covered_goal_ids=["goal-orders"],
                row_count=20,
            ),
            QueryOutcome(
                status=QueryOutcomeStatus.NEEDS_REASONING,
                query_id="refunds",
                observation=StructuredQueryObservation(
                    stage="CONTRACT",
                    code="TIME_FIELD_REQUIRED",
                    read_next=[{"refId": "semantic:refunds:field:refund_time"}],
                ),
            ),
        ],
        verified_count=1,
        needs_reasoning_count=1,
    )

    payload = outcome.to_core_observation().model_dump(
        by_alias=True,
        mode="json",
    )

    assert payload["decisionMode"] == "CALLER_REACT"
    assert payload["verifiedCount"] == 1
    assert payload["needsReasoningCount"] == 1
    assert [item["queryId"] for item in payload["branches"]] == [
        "orders",
        "refunds",
    ]
    assert payload["branches"] == payload["outcomes"]
    assert payload["branches"][0]["artifactIds"] == ["artifact-orders"]
    assert payload["branches"][1]["observation"]["readNext"] == [
        {"refId": "semantic:refunds:field:refund_time"}
    ]
