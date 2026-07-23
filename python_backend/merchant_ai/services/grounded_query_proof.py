from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from pydantic import Field
import sqlglot

from merchant_ai.models import APIModel, AgentRunResult, QueryPlan
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_sql_candidate import (
    GroundedSqlValidationResult,
    grounded_query_contract_fingerprint,
)


class GroundedQueryProofBundle(APIModel):
    """Immutable authority passed from governed execution to evidence checks.

    ``execution_plan`` remains a compatibility projection while legacy evidence
    contracts are migrated. It is sealed inside the proof and is no longer the
    public authority used to identify the executed query.
    """

    proof_version: str = "grounded_query_proof.v1"
    question: str
    contract: GroundedQueryContract
    execution_plan: QueryPlan = Field(exclude=True, repr=False)
    run_result: AgentRunResult = Field(exclude=True, repr=False)
    contract_fingerprint: str
    sql_ast_fingerprint: str
    candidate_sql_ast_fingerprint: str = ""
    execution_plan_fingerprint: str
    run_result_fingerprint: str
    merchant_scope_fingerprint: str
    semantic_activation_fingerprint: str = ""
    physical_plan_fingerprint: str = ""
    sql_validation: Optional[GroundedSqlValidationResult] = None
    proof_fingerprint: str = ""

    def fingerprint_valid(self) -> bool:
        return bool(
            self.proof_fingerprint
            and self.proof_fingerprint == grounded_query_proof_fingerprint(self)
        )


def build_grounded_query_proof(
    *,
    question: str,
    contract: GroundedQueryContract,
    execution_plan: QueryPlan,
    run_result: AgentRunResult,
    merchant_scope_fingerprint: str,
    semantic_activation_fingerprint: str = "",
    sql_validation: GroundedSqlValidationResult | None = None,
) -> GroundedQueryProofBundle:
    contract_copy = contract.model_copy(deep=True)
    plan_copy = execution_plan.model_copy(deep=True)
    result_copy = run_result.model_copy(deep=True)
    validation_copy = (
        sql_validation.model_copy(deep=True)
        if sql_validation is not None
        else None
    )
    contract_fingerprint = grounded_query_contract_fingerprint(contract_copy)
    sql_ast_fingerprint = _executed_sql_ast_fingerprint(
        result_copy.merged_query_bundle.sql
    )
    physical_plan = dict(result_copy.physical_plan_assessment or {})
    proof = GroundedQueryProofBundle(
        question=str(question or "").strip(),
        contract=contract_copy,
        execution_plan=plan_copy,
        run_result=result_copy,
        contract_fingerprint=contract_fingerprint,
        sql_ast_fingerprint=sql_ast_fingerprint,
        candidate_sql_ast_fingerprint=(
            str(validation_copy.ast_fingerprint or "")
            if validation_copy is not None
            else ""
        ),
        execution_plan_fingerprint=_stable_json_hash(
            plan_copy.model_dump(by_alias=True, mode="json")
        ),
        run_result_fingerprint=_authority_run_result_fingerprint(result_copy),
        merchant_scope_fingerprint=str(merchant_scope_fingerprint or "").strip(),
        semantic_activation_fingerprint=str(
            semantic_activation_fingerprint or ""
        ).strip(),
        physical_plan_fingerprint=(
            _stable_json_hash(physical_plan) if physical_plan else ""
        ),
        sql_validation=validation_copy,
    )
    return proof.model_copy(
        update={"proof_fingerprint": grounded_query_proof_fingerprint(proof)},
        deep=True,
    )


def grounded_query_proof_fingerprint(proof: GroundedQueryProofBundle) -> str:
    return _stable_json_hash(
        {
            "proofVersion": proof.proof_version,
            "question": proof.question,
            "contractFingerprint": proof.contract_fingerprint,
            "sqlAstFingerprint": proof.sql_ast_fingerprint,
            "candidateSqlAstFingerprint": (
                proof.candidate_sql_ast_fingerprint
            ),
            "executionPlanFingerprint": proof.execution_plan_fingerprint,
            "runResultFingerprint": proof.run_result_fingerprint,
            "merchantScopeFingerprint": proof.merchant_scope_fingerprint,
            "semanticActivationFingerprint": (
                proof.semantic_activation_fingerprint
            ),
            "physicalPlanFingerprint": proof.physical_plan_fingerprint,
            "sqlValidation": (
                proof.sql_validation.model_dump(by_alias=True, mode="json")
                if proof.sql_validation is not None
                else None
            ),
        }
    )


def grounded_query_proof_integrity_errors(
    proof: GroundedQueryProofBundle,
) -> list[tuple[str, str]]:
    errors: list[tuple[str, str]] = []
    observed_contract = grounded_query_contract_fingerprint(proof.contract)
    if proof.question.strip() != proof.contract.question.strip():
        errors.append(
            (
                "QUERY_PROOF_QUESTION_MISMATCH",
                "query proof question does not match the sealed Contract",
            )
        )
    if proof.contract_fingerprint != observed_contract:
        errors.append(
            (
                "QUERY_PROOF_CONTRACT_MISMATCH",
                "query proof Contract fingerprint does not match its payload",
            )
        )
    observed_plan = _stable_json_hash(
        proof.execution_plan.model_dump(by_alias=True, mode="json")
    )
    if proof.execution_plan_fingerprint != observed_plan:
        errors.append(
            (
                "QUERY_PROOF_PLAN_MISMATCH",
                "query proof execution projection changed after sealing",
            )
        )
    observed_result = _authority_run_result_fingerprint(proof.run_result)
    if proof.run_result_fingerprint != observed_result:
        errors.append(
            (
                "QUERY_PROOF_RESULT_MISMATCH",
                "query proof result authority changed after sealing",
            )
        )
    observed_sql = _executed_sql_ast_fingerprint(
        proof.run_result.merged_query_bundle.sql
    )
    if proof.sql_ast_fingerprint != observed_sql:
        errors.append(
            (
                "QUERY_PROOF_SQL_MISMATCH",
                "query proof SQL fingerprint does not match the executed SQL",
            )
        )
    validation = proof.sql_validation
    if validation is not None and (
        not validation.valid
        or validation.contract_fingerprint != proof.contract_fingerprint
        or validation.ast_fingerprint
        != proof.candidate_sql_ast_fingerprint
    ):
        errors.append(
            (
                "QUERY_PROOF_SQL_VALIDATION_MISMATCH",
                "SQL validation is not valid for the sealed Contract and SQL",
            )
        )
    physical_plan = dict(proof.run_result.physical_plan_assessment or {})
    observed_physical = _stable_json_hash(physical_plan) if physical_plan else ""
    if proof.physical_plan_fingerprint != observed_physical:
        errors.append(
            (
                "QUERY_PROOF_PHYSICAL_PLAN_MISMATCH",
                "physical-plan assessment changed after proof sealing",
            )
        )
    if not proof.merchant_scope_fingerprint:
        errors.append(
            (
                "QUERY_PROOF_SCOPE_MISSING",
                "query proof is not bound to a merchant authorization scope",
            )
        )
    snapshot_fingerprints = {
        str(bundle.data_snapshot.semantic_activation_fingerprint or "").strip()
        for bundle in [
            proof.run_result.merged_query_bundle,
            *proof.run_result.query_bundles,
        ]
        if str(bundle.data_snapshot.semantic_activation_fingerprint or "").strip()
    }
    if proof.semantic_activation_fingerprint:
        if not snapshot_fingerprints:
            errors.append(
                (
                    "QUERY_PROOF_SEMANTIC_ACTIVATION_MISSING",
                    "result snapshot is not bound to the active semantic version",
                )
            )
        elif snapshot_fingerprints != {
            proof.semantic_activation_fingerprint
        }:
            errors.append(
                (
                    "QUERY_PROOF_SEMANTIC_ACTIVATION_MISMATCH",
                    "result snapshot is bound to a different semantic activation",
                )
            )
    if not proof.fingerprint_valid():
        errors.append(
            (
                "QUERY_PROOF_FINGERPRINT_INVALID",
                "query proof fingerprint is missing or invalid",
            )
        )
    return errors


def _executed_sql_ast_fingerprint(sql: str) -> str:
    source = str(sql or "").strip()
    if not source:
        return hashlib.sha256(b"").hexdigest()
    expression = None
    for dialect in ("doris", "mysql"):
        try:
            expression = sqlglot.parse_one(source, read=dialect)
            break
        except Exception:
            continue
    if expression is None:
        return hashlib.sha256(source.encode("utf-8")).hexdigest()
    structural = [
        {key: value for key, value in item.items() if key != "m"}
        for item in expression.dump()
    ]
    return _stable_json_hash(structural)


def _authority_run_result_fingerprint(run_result: AgentRunResult) -> str:
    payload = run_result.model_dump(by_alias=True, mode="json")
    for key in (
        "verifiedEvidence",
        "verifiedFacts",
        "answerClaimVerification",
    ):
        payload.pop(key, None)
    return _stable_json_hash(payload)


def _stable_json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
