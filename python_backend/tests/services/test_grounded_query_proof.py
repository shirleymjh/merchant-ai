from __future__ import annotations

from merchant_ai.models import (
    AgentRunResult,
    AgentTaskResult,
    QueryBundle,
    QueryPlan,
    ResultCoverage,
)
from merchant_ai.services.evidence import EvidenceVerifier
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedRankingBinding,
)
from merchant_ai.services.grounded_query_proof import build_grounded_query_proof
from merchant_ai.services.grounded_semantic_ir import GroundedOutputProjection
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)


QUESTION = "最近 7 天是否有退款明细？"
TABLE = "refund_detail"
OUTPUT = "refund_id"
SQL = "SELECT refund_id FROM refund_detail WHERE merchant_id = 'm1'"


def _contract() -> GroundedQueryContract:
    return GroundedQueryContract(
        status="READY",
        question=QUESTION,
        query_shape="DETAIL",
        primary_table=TABLE,
        requested_outputs=[
            GroundedOutputProjection(
                semantic_ref_id="semantic:refund_id",
                output_alias=OUTPUT,
                binding_kind="FIELD",
            )
        ],
    )


def _plan() -> QueryPlan:
    return QueryPlan(
        evidence_contracts=[
            {
                "taskId": "refunds",
                "table": TABLE,
                "columns": [OUTPUT],
                "semanticLabel": "refund detail population",
            }
        ]
    )


def _run(*, coverage: ResultCoverage) -> AgentRunResult:
    bundle = QueryBundle(
        sql=SQL,
        tables=[TABLE],
        rows=[],
        original_row_count=0,
        result_coverage=coverage,
        is_truncated=coverage == ResultCoverage.PREVIEW,
    )
    return AgentRunResult(
        task_results=[
            AgentTaskResult(
                task_id="refunds",
                success=True,
                query_bundle=bundle.model_copy(deep=True),
            )
        ],
        query_bundles=[bundle.model_copy(deep=True)],
        merged_query_bundle=bundle.model_copy(deep=True),
    )


def _proof(*, coverage: ResultCoverage = ResultCoverage.ALL_ROWS):
    contract = _contract()
    validation = GroundedSqlValidationResult(
        valid=True,
        canonical_sql=SQL,
        ast_fingerprint="sql-ast-fingerprint",
        contract_fingerprint=grounded_query_contract_fingerprint(contract),
        output_columns=[OUTPUT],
    )
    return build_grounded_query_proof(
        question=QUESTION,
        contract=contract,
        execution_plan=_plan(),
        run_result=_run(coverage=coverage),
        merchant_scope_fingerprint="merchant-scope-m1",
        semantic_activation_fingerprint="",
        sql_validation=validation,
    )


def test_complete_empty_population_is_valid_evidence() -> None:
    proof = _proof()
    verified = EvidenceVerifier().verify_proof(proof)

    assert verified.passed is True
    assert proof.candidate_sql_ast_fingerprint == "sql-ast-fingerprint"
    assert proof.sql_ast_fingerprint != proof.candidate_sql_ast_fingerprint
    assert "refund detail population" in verified.covered_evidence
    assert "EMPTY_RESULT_COVERAGE_UNPROVEN" not in {
        gap.code for gap in verified.gaps
    }


def test_empty_preview_cannot_prove_that_no_records_exist() -> None:
    verified = EvidenceVerifier().verify_proof(
        _proof(coverage=ResultCoverage.PREVIEW)
    )

    assert verified.passed is False
    assert {
        "EMPTY_RESULT_COVERAGE_UNPROVEN",
        "RESULT_PREVIEW_ONLY",
    } <= {gap.code for gap in verified.gaps}


def test_query_proof_tampering_fails_closed() -> None:
    proof = _proof().model_copy(
        update={"merchant_scope_fingerprint": ""},
        deep=True,
    )

    verified = EvidenceVerifier().verify_proof(proof)

    assert verified.passed is False
    assert {
        "QUERY_PROOF_SCOPE_MISSING",
        "QUERY_PROOF_FINGERPRINT_INVALID",
    } <= {gap.code for gap in verified.blocking_gaps}


def test_physical_plan_receipt_must_bind_executed_sql() -> None:
    proof = _proof()
    proof.run_result.physical_plan_assessment = {
        "assessmentId": "physical-1",
        "status": "VERIFIED",
        "executable": True,
        "sqlFingerprint": "wrong-sql",
        "gaps": [],
    }
    proof = build_grounded_query_proof(
        question=proof.question,
        contract=proof.contract,
        execution_plan=proof.execution_plan,
        run_result=proof.run_result,
        merchant_scope_fingerprint=proof.merchant_scope_fingerprint,
        sql_validation=proof.sql_validation,
    )

    verified = EvidenceVerifier().verify_proof(proof)

    assert verified.passed is False
    assert "PHYSICAL_PLAN_SQL_FINGERPRINT_MISMATCH" in {
        gap.code for gap in verified.blocking_gaps
    }


def test_grouped_preview_cannot_support_complete_aggregate_conclusion() -> None:
    proof = _proof(coverage=ResultCoverage.PREVIEW)
    proof.contract.query_shape = "GROUPED"
    proof = build_grounded_query_proof(
        question=proof.question,
        contract=proof.contract,
        execution_plan=proof.execution_plan,
        run_result=proof.run_result,
        merchant_scope_fingerprint=proof.merchant_scope_fingerprint,
        sql_validation=proof.sql_validation.model_copy(
            update={
                "contract_fingerprint": grounded_query_contract_fingerprint(
                    proof.contract
                )
            }
        ),
    )

    verified = EvidenceVerifier().verify_proof(proof)

    assert verified.passed is False
    assert "RESULT_SET_COVERAGE_INCOMPLETE" in {
        gap.code for gap in verified.blocking_gaps
    }


def test_ranked_result_cannot_exceed_contract_limit() -> None:
    proof = _proof()
    proof.contract.query_shape = "RANKED"
    proof.contract.ranking = GroundedRankingBinding(
        enabled=True,
        direction="DESC",
        limit=2,
    )
    rows = [{OUTPUT: "r1"}, {OUTPUT: "r2"}, {OUTPUT: "r3"}]
    proof.run_result.merged_query_bundle.rows = rows
    proof.run_result.merged_query_bundle.original_row_count = len(rows)
    proof.run_result.merged_query_bundle.result_coverage = ResultCoverage.TOP_N
    proof.run_result.task_results[0].query_bundle.rows = rows
    proof.run_result.task_results[0].query_bundle.result_coverage = (
        ResultCoverage.TOP_N
    )
    proof = build_grounded_query_proof(
        question=proof.question,
        contract=proof.contract,
        execution_plan=proof.execution_plan,
        run_result=proof.run_result,
        merchant_scope_fingerprint=proof.merchant_scope_fingerprint,
        sql_validation=proof.sql_validation.model_copy(
            update={
                "contract_fingerprint": grounded_query_contract_fingerprint(
                    proof.contract
                )
            }
        ),
    )

    verified = EvidenceVerifier().verify_proof(proof)

    assert verified.passed is False
    assert "RANKING_RESULT_LIMIT_MISMATCH" in {
        gap.code for gap in verified.blocking_gaps
    }
