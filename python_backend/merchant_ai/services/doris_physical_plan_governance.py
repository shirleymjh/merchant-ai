from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import sqlglot
from pydantic import Field
from sqlglot import exp
from sqlglot.optimizer.scope import Scope, traverse_scope

from merchant_ai.models import APIModel


class DorisInvertedIndexContract(APIModel):
    name: str = ""
    column: str
    index_type: str = "INVERTED"


class DorisPhysicalTableContract(APIModel):
    """Physical capabilities copied from one published formal table asset.

    These fields describe storage layout only.  In particular, a partition
    column is never treated as a business-time field by this contract.
    """

    table: str
    topic: str = ""
    asset_ref: str = ""
    asset_status: str = ""
    partition_columns: list[str] = Field(default_factory=list)
    partition_count: int = 0
    bucket_columns: list[str] = Field(default_factory=list)
    bucket_count: int = 0
    colocate_group: str = ""
    inverted_indexes: list[DorisInvertedIndexContract] = Field(default_factory=list)
    metadata_fingerprint: str = ""


class PartitionPruningRequirement(APIModel):
    """A semantic Contract's explicit request for physical partition pruning."""

    table: str
    partition_column: str
    semantic_evidence_ref: str
    blocking: bool = True


class PhysicalPlanGovernancePolicy(APIModel):
    bucket_filter_blocking: bool = False
    bucket_join_blocking: bool = False
    colocate_join_blocking: bool = False
    inverted_index_blocking: bool = False


class PhysicalPlanEvidence(APIModel):
    source: Literal["FORMAL_ASSET", "SQL_AST", "EXPLAIN_VERBOSE"]
    locator: str = ""
    excerpt: str = ""
    attributes: dict[str, Any] = Field(default_factory=dict)


class PhysicalPlanGap(APIModel):
    code: str
    message: str
    obligation_id: str = ""
    kind: str = ""
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    blocking: bool = False
    expected_evidence: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class PhysicalPlanObligationAssessment(APIModel):
    obligation_id: str
    kind: Literal[
        "PARTITION_PRUNING",
        "BUCKET_FILTER_PRUNING",
        "BUCKET_JOIN_ALIGNMENT",
        "COLOCATE_JOIN",
        "INVERTED_INDEX_USAGE",
    ]
    status: Literal["VERIFIED", "GAP"]
    tables: list[str] = Field(default_factory=list)
    columns: list[str] = Field(default_factory=list)
    blocking: bool = False
    source: Literal["SEMANTIC_CONTRACT", "FORMAL_ASSET_SQL_AST"]
    expected_evidence: str = ""
    evidence: list[PhysicalPlanEvidence] = Field(default_factory=list)
    gap_codes: list[str] = Field(default_factory=list)
    capability: dict[str, Any] = Field(default_factory=dict)


class PhysicalPlanAssessment(APIModel):
    assessment_version: str = "doris_physical_plan_assessment.v1"
    assessment_id: str = ""
    status: Literal["VERIFIED", "GAPPED", "NOT_REQUIRED", "INVALID"] = "INVALID"
    executable: bool = False
    sql_fingerprint: str = ""
    explain_fingerprint: str = ""
    policy_fingerprint: str = ""
    referenced_tables: list[str] = Field(default_factory=list)
    formal_assets: list[DorisPhysicalTableContract] = Field(default_factory=list)
    obligations: list[PhysicalPlanObligationAssessment] = Field(default_factory=list)
    gaps: list[PhysicalPlanGap] = Field(default_factory=list)
    sql_was_rewritten: bool = False


@dataclass(frozen=True)
class _ExplainLine:
    locator: str
    field: str
    text: str
    tokens: tuple[str, ...]
    direct_table_refs: tuple[str, ...] = ()
    table_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ParsedSql:
    expression: exp.Expression
    referenced_tables: tuple[str, ...]
    filter_columns: frozenset[tuple[str, str]]
    equality_filter_columns: frozenset[tuple[str, str]]
    join_pairs: frozenset[tuple[tuple[str, str], tuple[str, str]]]
    has_join: bool
    lineage_gaps: tuple[PhysicalPlanGap, ...] = ()


@dataclass(frozen=True)
class _ScopedSource:
    alias: str
    base_table: str = ""
    outputs: Mapping[str, frozenset[tuple[str, str]]] | None = None
    base_tables: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _ScopedSqlState:
    sources: Mapping[str, _ScopedSource]
    outputs: Mapping[str, frozenset[tuple[str, str]]]
    output_order: tuple[str, ...]
    base_tables: frozenset[str]


class DorisPhysicalPlanGovernor:
    """Validate physical-plan evidence without planning or rewriting SQL."""

    def assess(
        self,
        *,
        sql: str,
        formal_assets: Sequence[Mapping[str, Any] | APIModel],
        explain_payload: Any,
        partition_requirements: Sequence[PartitionPruningRequirement] = (),
        policy: PhysicalPlanGovernancePolicy | None = None,
    ) -> PhysicalPlanAssessment:
        source_sql = str(sql or "").strip()
        sql_fingerprint = _fingerprint_text(source_sql)
        explain_fingerprint = _fingerprint_json(explain_payload)
        active_policy = policy or PhysicalPlanGovernancePolicy()
        policy_fingerprint = _fingerprint_json(
            {
                "policy": active_policy.model_dump(by_alias=True, mode="json"),
                "partitionRequirements": [
                    item.model_dump(by_alias=True, mode="json")
                    for item in partition_requirements
                ],
            }
        )
        base = {
            "sql_fingerprint": sql_fingerprint,
            "explain_fingerprint": explain_fingerprint,
            "policy_fingerprint": policy_fingerprint,
            "sql_was_rewritten": False,
        }
        try:
            expression = _parse_single_read_query(source_sql)
        except ValueError as exc:
            gap = PhysicalPlanGap(
                code="PHYSICAL_PLAN_SQL_INVALID",
                message=str(exc),
                blocking=True,
                expected_evidence="one read-only Doris query",
            )
            return _final_assessment(base=base, gaps=[gap], invalid=True)

        referenced_tables = _referenced_base_tables(expression)
        contracts, asset_gaps = _formal_asset_contracts(
            formal_assets,
            referenced_tables,
        )
        base["referenced_tables"] = list(referenced_tables)
        base["formal_assets"] = list(contracts.values())
        if asset_gaps:
            return _final_assessment(base=base, gaps=asset_gaps, invalid=True)

        parsed = _inspect_sql(expression, contracts)
        explain_lines = _explain_lines(
            explain_payload,
            parsed.referenced_tables,
        )
        obligations: list[PhysicalPlanObligationAssessment] = []
        gaps: list[PhysicalPlanGap] = list(parsed.lineage_gaps)

        for requirement in partition_requirements:
            obligation, obligation_gaps = self._assess_partition_requirement(
                requirement,
                contracts,
                parsed,
                explain_lines,
            )
            obligations.append(obligation)
            gaps.extend(obligation_gaps)

        bucket_obligations, bucket_gaps = self._assess_bucket_filters(
            contracts,
            parsed,
            explain_lines,
            active_policy,
        )
        obligations.extend(bucket_obligations)
        gaps.extend(bucket_gaps)

        join_obligations, join_gaps = self._assess_bucket_joins(
            contracts,
            parsed,
            explain_lines,
            active_policy,
        )
        obligations.extend(join_obligations)
        gaps.extend(join_gaps)

        index_obligations, index_gaps = self._assess_inverted_indexes(
            contracts,
            parsed,
            explain_lines,
            active_policy,
        )
        obligations.extend(index_obligations)
        gaps.extend(index_gaps)

        base["obligations"] = obligations
        return _final_assessment(base=base, gaps=gaps)

    def _assess_partition_requirement(
        self,
        requirement: PartitionPruningRequirement,
        contracts: Mapping[str, DorisPhysicalTableContract],
        parsed: _ParsedSql,
        lines: Sequence[_ExplainLine],
    ) -> tuple[PhysicalPlanObligationAssessment, list[PhysicalPlanGap]]:
        table_key = _key(requirement.table)
        column_key = _key(requirement.partition_column)
        contract = contracts.get(table_key)
        table = contract.table if contract else str(requirement.table or "").strip()
        column = str(requirement.partition_column or "").strip()
        obligation_id = _obligation_id(
            "PARTITION_PRUNING",
            [table],
            [column],
            requirement.semantic_evidence_ref,
        )
        expected = "EXPLAIN VERBOSE must prove a reduced partition scan for the Contract-declared pruning field"
        evidence: list[PhysicalPlanEvidence] = []
        local_gaps: list[PhysicalPlanGap] = []

        def add_gap(code: str, message: str, details: Mapping[str, Any] | None = None) -> None:
            local_gaps.append(
                PhysicalPlanGap(
                    code=code,
                    message=message,
                    obligation_id=obligation_id,
                    kind="PARTITION_PRUNING",
                    tables=[table] if table else [],
                    columns=[column] if column else [],
                    blocking=requirement.blocking,
                    expected_evidence=expected,
                    details=dict(details or {}),
                )
            )

        if not str(requirement.semantic_evidence_ref or "").strip():
            add_gap(
                "PARTITION_REQUIREMENT_SEMANTIC_EVIDENCE_MISSING",
                "Partition pruning cannot be required without a semantic Contract evidence reference",
            )
        if contract is None:
            add_gap(
                "PARTITION_REQUIREMENT_TABLE_UNKNOWN",
                "The pruning requirement targets no referenced published formal asset",
            )
        elif column_key not in {_key(item) for item in contract.partition_columns}:
            add_gap(
                "PARTITION_COLUMN_NOT_DECLARED",
                "The semantic Contract's pruning field is not declared by formal physicalMetadata",
                {"declaredPartitionColumns": list(contract.partition_columns)},
            )
        else:
            evidence.append(
                _asset_evidence(
                    contract,
                    {"partitionColumn": column},
                )
            )
            if (table_key, column_key) not in parsed.filter_columns:
                add_gap(
                    "PARTITION_FILTER_ABSENT",
                    "The SQL AST contains no filter on the Contract-declared pruning field",
                )
            else:
                evidence.append(
                    PhysicalPlanEvidence(
                        source="SQL_AST",
                        locator="filter",
                        attributes={"table": table, "column": column},
                    )
                )
                plan_evidence, plan_code, plan_details = _ratio_plan_evidence(
                    lines,
                    table,
                    kind="partition",
                    known_total=contract.partition_count,
                    referenced_table_count=len(parsed.referenced_tables),
                )
                if plan_evidence is not None:
                    evidence.append(plan_evidence)
                else:
                    add_gap(
                        plan_code,
                        (
                            "EXPLAIN VERBOSE reports no effective partition reduction"
                            if plan_code == "PARTITION_SCAN_NOT_REDUCED"
                            else "EXPLAIN VERBOSE contains no table-scoped partition-pruning proof"
                        ),
                        plan_details,
                    )

        gap_codes = [item.code for item in local_gaps]
        return (
            PhysicalPlanObligationAssessment(
                obligation_id=obligation_id,
                kind="PARTITION_PRUNING",
                status="GAP" if local_gaps else "VERIFIED",
                tables=[table] if table else [],
                columns=[column] if column else [],
                blocking=requirement.blocking,
                source="SEMANTIC_CONTRACT",
                expected_evidence=expected,
                evidence=evidence,
                gap_codes=gap_codes,
                capability={
                    "semanticEvidenceRef": requirement.semantic_evidence_ref,
                    "businessTimeInferredFromPartition": False,
                },
            ),
            local_gaps,
        )

    def _assess_bucket_filters(
        self,
        contracts: Mapping[str, DorisPhysicalTableContract],
        parsed: _ParsedSql,
        lines: Sequence[_ExplainLine],
        policy: PhysicalPlanGovernancePolicy,
    ) -> tuple[list[PhysicalPlanObligationAssessment], list[PhysicalPlanGap]]:
        obligations: list[PhysicalPlanObligationAssessment] = []
        gaps: list[PhysicalPlanGap] = []
        for table_key in parsed.referenced_tables:
            contract = contracts[table_key]
            bucket_keys = [_key(item) for item in contract.bucket_columns]
            if not bucket_keys:
                continue
            constrained = [
                column
                for column in bucket_keys
                if (table_key, column) in parsed.equality_filter_columns
            ]
            if not constrained:
                continue
            columns = list(contract.bucket_columns)
            obligation_id = _obligation_id(
                "BUCKET_FILTER_PRUNING",
                [contract.table],
                columns,
            )
            expected = "all declared bucket columns constrained by equality and EXPLAIN VERBOSE tablet reduction"
            evidence = [
                _asset_evidence(
                    contract,
                    {
                        "bucketColumns": columns,
                        "bucketCount": contract.bucket_count,
                    },
                ),
                PhysicalPlanEvidence(
                    source="SQL_AST",
                    locator="equality_filter",
                    attributes={"constrainedBucketColumns": constrained},
                ),
            ]
            local_gaps: list[PhysicalPlanGap] = []
            if len(constrained) != len(bucket_keys):
                local_gaps.append(
                    _gap(
                        code="BUCKET_FILTER_PARTIAL",
                        message="The equality filter does not constrain every declared bucket column",
                        obligation_id=obligation_id,
                        kind="BUCKET_FILTER_PRUNING",
                        tables=[contract.table],
                        columns=columns,
                        blocking=policy.bucket_filter_blocking,
                        expected=expected,
                        details={"constrainedBucketColumns": constrained},
                    )
                )
            else:
                plan_evidence, plan_code, plan_details = _ratio_plan_evidence(
                    lines,
                    contract.table,
                    kind="tablet",
                    known_total=contract.bucket_count,
                    referenced_table_count=len(parsed.referenced_tables),
                )
                if plan_evidence is not None:
                    evidence.append(plan_evidence)
                else:
                    local_gaps.append(
                        _gap(
                            code=plan_code,
                            message=(
                                "EXPLAIN VERBOSE reports no effective tablet reduction"
                                if plan_code == "TABLET_SCAN_NOT_REDUCED"
                                else "EXPLAIN VERBOSE contains no table-scoped bucket/tablet pruning proof"
                            ),
                            obligation_id=obligation_id,
                            kind="BUCKET_FILTER_PRUNING",
                            tables=[contract.table],
                            columns=columns,
                            blocking=policy.bucket_filter_blocking,
                            expected=expected,
                            details=plan_details,
                        )
                    )
            obligations.append(
                PhysicalPlanObligationAssessment(
                    obligation_id=obligation_id,
                    kind="BUCKET_FILTER_PRUNING",
                    status="GAP" if local_gaps else "VERIFIED",
                    tables=[contract.table],
                    columns=columns,
                    blocking=policy.bucket_filter_blocking,
                    source="FORMAL_ASSET_SQL_AST",
                    expected_evidence=expected,
                    evidence=evidence,
                    gap_codes=[item.code for item in local_gaps],
                    capability={"bucketCount": contract.bucket_count},
                )
            )
            gaps.extend(local_gaps)
        return obligations, gaps

    def _assess_bucket_joins(
        self,
        contracts: Mapping[str, DorisPhysicalTableContract],
        parsed: _ParsedSql,
        lines: Sequence[_ExplainLine],
        policy: PhysicalPlanGovernancePolicy,
    ) -> tuple[list[PhysicalPlanObligationAssessment], list[PhysicalPlanGap]]:
        obligations: list[PhysicalPlanObligationAssessment] = []
        gaps: list[PhysicalPlanGap] = []
        table_pairs = sorted(
            {
                tuple(sorted((left[0], right[0])))
                for left, right in parsed.join_pairs
                if left[0] != right[0]
            }
        )
        if parsed.has_join and not table_pairs:
            bucketed = [
                contracts[table_key]
                for table_key in parsed.referenced_tables
                if contracts[table_key].bucket_columns
            ]
            if len(bucketed) >= 2:
                tables = [item.table for item in bucketed]
                columns = [
                    "%s.%s" % (item.table, column)
                    for item in bucketed
                    for column in item.bucket_columns
                ]
                obligation_id = _obligation_id(
                    "BUCKET_JOIN_ALIGNMENT",
                    tables,
                    columns,
                )
                expected = "a resolvable cross-table equality lineage over declared distribution keys"
                gap = _gap(
                    code="BUCKET_JOIN_LINEAGE_UNRESOLVED",
                    message="The SQL contains a join, but its base-table bucket-key equality lineage cannot be proven",
                    obligation_id=obligation_id,
                    kind="BUCKET_JOIN_ALIGNMENT",
                    tables=tables,
                    columns=columns,
                    blocking=policy.bucket_join_blocking,
                    expected=expected,
                )
                obligations.append(
                    PhysicalPlanObligationAssessment(
                        obligation_id=obligation_id,
                        kind="BUCKET_JOIN_ALIGNMENT",
                        status="GAP",
                        tables=tables,
                        columns=columns,
                        blocking=policy.bucket_join_blocking,
                        source="FORMAL_ASSET_SQL_AST",
                        expected_evidence=expected,
                        evidence=[
                            _asset_evidence(
                                item,
                                {"bucketColumns": item.bucket_columns},
                            )
                            for item in bucketed
                        ],
                        gap_codes=[gap.code],
                        capability={"joinLineageResolved": False},
                    )
                )
                gaps.append(gap)
            return obligations, gaps
        for left_key, right_key in table_pairs:
            left = contracts[left_key]
            right = contracts[right_key]
            if not left.bucket_columns and not right.bucket_columns:
                continue
            columns = [
                *["%s.%s" % (left.table, item) for item in left.bucket_columns],
                *["%s.%s" % (right.table, item) for item in right.bucket_columns],
            ]
            obligation_id = _obligation_id(
                "BUCKET_JOIN_ALIGNMENT",
                [left.table, right.table],
                columns,
            )
            expected = "join equality must cover corresponding declared distribution keys"
            expected_pairs = _expected_bucket_join_pairs(left, right)
            actual_pairs = {
                _ordered_join_pair(pair)
                for pair in parsed.join_pairs
                if {pair[0][0], pair[1][0]} == {left_key, right_key}
            }
            aligned = bool(expected_pairs) and expected_pairs.issubset(actual_pairs)
            local_gaps: list[PhysicalPlanGap] = []
            if not left.bucket_columns or not right.bucket_columns:
                local_gaps.append(
                    _gap(
                        code="BUCKET_JOIN_CAPABILITY_INCOMPLETE",
                        message="Only one side of the join declares bucket capabilities",
                        obligation_id=obligation_id,
                        kind="BUCKET_JOIN_ALIGNMENT",
                        tables=[left.table, right.table],
                        columns=columns,
                        blocking=policy.bucket_join_blocking,
                        expected=expected,
                    )
                )
            elif len(left.bucket_columns) != len(right.bucket_columns):
                local_gaps.append(
                    _gap(
                        code="BUCKET_KEY_ARITY_MISMATCH",
                        message="Joined tables declare different numbers of bucket columns",
                        obligation_id=obligation_id,
                        kind="BUCKET_JOIN_ALIGNMENT",
                        tables=[left.table, right.table],
                        columns=columns,
                        blocking=policy.bucket_join_blocking,
                        expected=expected,
                    )
                )
            elif not aligned:
                local_gaps.append(
                    _gap(
                        code="BUCKET_JOIN_NOT_ALIGNED",
                        message="Join equality does not cover corresponding declared bucket columns",
                        obligation_id=obligation_id,
                        kind="BUCKET_JOIN_ALIGNMENT",
                        tables=[left.table, right.table],
                        columns=columns,
                        blocking=policy.bucket_join_blocking,
                        expected=expected,
                        details={
                            "expectedPairs": [list(item) for item in sorted(expected_pairs)],
                            "actualPairs": [list(item) for item in sorted(actual_pairs)],
                        },
                    )
                )
            obligations.append(
                PhysicalPlanObligationAssessment(
                    obligation_id=obligation_id,
                    kind="BUCKET_JOIN_ALIGNMENT",
                    status="GAP" if local_gaps else "VERIFIED",
                    tables=[left.table, right.table],
                    columns=columns,
                    blocking=policy.bucket_join_blocking,
                    source="FORMAL_ASSET_SQL_AST",
                    expected_evidence=expected,
                    evidence=[
                        _asset_evidence(left, {"bucketColumns": left.bucket_columns}),
                        _asset_evidence(right, {"bucketColumns": right.bucket_columns}),
                        PhysicalPlanEvidence(
                            source="SQL_AST",
                            locator="join_equality",
                            attributes={
                                "actualPairs": [list(item) for item in sorted(actual_pairs)]
                            },
                        ),
                    ],
                    gap_codes=[item.code for item in local_gaps],
                    capability={"bucketAligned": aligned},
                )
            )
            gaps.extend(local_gaps)

            if not aligned:
                continue
            colocate_declared = bool(
                left.colocate_group
                and right.colocate_group
                and left.colocate_group == right.colocate_group
            )
            bucket_counts_match = bool(
                left.bucket_count
                and right.bucket_count
                and left.bucket_count == right.bucket_count
            )
            if not colocate_declared:
                continue
            colocate_id = _obligation_id(
                "COLOCATE_JOIN",
                [left.table, right.table],
                columns,
                left.colocate_group,
            )
            colocate_expected = "EXPLAIN VERBOSE must affirm colocate execution for a formally declared colocate-capable join"
            colocate_evidence = [
                _asset_evidence(
                    left,
                    {
                        "colocateGroup": left.colocate_group,
                        "bucketCount": left.bucket_count,
                    },
                ),
                _asset_evidence(
                    right,
                    {
                        "colocateGroup": right.colocate_group,
                        "bucketCount": right.bucket_count,
                    },
                ),
            ]
            colocate_gaps: list[PhysicalPlanGap] = []
            if not bucket_counts_match:
                colocate_gaps.append(
                    _gap(
                        code="COLOCATE_BUCKET_COUNT_MISMATCH",
                        message="Tables in the declared colocate group have incompatible bucket counts",
                        obligation_id=colocate_id,
                        kind="COLOCATE_JOIN",
                        tables=[left.table, right.table],
                        columns=columns,
                        blocking=policy.colocate_join_blocking,
                        expected=colocate_expected,
                        details={
                            "leftBucketCount": left.bucket_count,
                            "rightBucketCount": right.bucket_count,
                        },
                    )
                )
            else:
                plan_evidence = _colocate_plan_evidence(
                    lines,
                    left.table,
                    right.table,
                )
                if plan_evidence is not None:
                    colocate_evidence.append(plan_evidence)
                else:
                    colocate_gaps.append(
                        _gap(
                            code="COLOCATE_PLAN_EVIDENCE_MISSING",
                            message="EXPLAIN VERBOSE does not affirm colocate execution",
                            obligation_id=colocate_id,
                            kind="COLOCATE_JOIN",
                            tables=[left.table, right.table],
                            columns=columns,
                            blocking=policy.colocate_join_blocking,
                            expected=colocate_expected,
                        )
                    )
            obligations.append(
                PhysicalPlanObligationAssessment(
                    obligation_id=colocate_id,
                    kind="COLOCATE_JOIN",
                    status="GAP" if colocate_gaps else "VERIFIED",
                    tables=[left.table, right.table],
                    columns=columns,
                    blocking=policy.colocate_join_blocking,
                    source="FORMAL_ASSET_SQL_AST",
                    expected_evidence=colocate_expected,
                    evidence=colocate_evidence,
                    gap_codes=[item.code for item in colocate_gaps],
                    capability={
                        "colocateGroup": left.colocate_group,
                        "bucketCountsMatch": bucket_counts_match,
                    },
                )
            )
            gaps.extend(colocate_gaps)
        return obligations, gaps

    def _assess_inverted_indexes(
        self,
        contracts: Mapping[str, DorisPhysicalTableContract],
        parsed: _ParsedSql,
        lines: Sequence[_ExplainLine],
        policy: PhysicalPlanGovernancePolicy,
    ) -> tuple[list[PhysicalPlanObligationAssessment], list[PhysicalPlanGap]]:
        obligations: list[PhysicalPlanObligationAssessment] = []
        gaps: list[PhysicalPlanGap] = []
        all_index_names = [
            _key(index.name)
            for contract in contracts.values()
            for index in contract.inverted_indexes
            if index.name
        ]
        for table_key in parsed.referenced_tables:
            contract = contracts[table_key]
            for index in contract.inverted_indexes:
                column_key = _key(index.column)
                if (table_key, column_key) not in parsed.filter_columns:
                    continue
                obligation_id = _obligation_id(
                    "INVERTED_INDEX_USAGE",
                    [contract.table],
                    [index.column],
                    index.name,
                )
                expected = "EXPLAIN VERBOSE must name the declared index, or affirm an inverted-index scan for its filtered column"
                evidence = [
                    _asset_evidence(
                        contract,
                        {
                            "indexName": index.name,
                            "indexColumn": index.column,
                            "indexType": index.index_type,
                        },
                    ),
                    PhysicalPlanEvidence(
                        source="SQL_AST",
                        locator="filter",
                        attributes={"table": contract.table, "column": index.column},
                    ),
                ]
                plan_evidence = _inverted_index_plan_evidence(
                    lines,
                    table=contract.table,
                    column=index.column,
                    index_name=index.name,
                    index_name_unique=(
                        bool(index.name)
                        and all_index_names.count(_key(index.name)) == 1
                    ),
                    referenced_table_count=len(parsed.referenced_tables),
                )
                local_gaps: list[PhysicalPlanGap] = []
                if plan_evidence is not None:
                    evidence.append(plan_evidence)
                else:
                    local_gaps.append(
                        _gap(
                            code="INVERTED_INDEX_PLAN_EVIDENCE_MISSING",
                            message="EXPLAIN VERBOSE does not prove use of the declared inverted index",
                            obligation_id=obligation_id,
                            kind="INVERTED_INDEX_USAGE",
                            tables=[contract.table],
                            columns=[index.column],
                            blocking=policy.inverted_index_blocking,
                            expected=expected,
                            details={"indexName": index.name},
                        )
                    )
                obligations.append(
                    PhysicalPlanObligationAssessment(
                        obligation_id=obligation_id,
                        kind="INVERTED_INDEX_USAGE",
                        status="GAP" if local_gaps else "VERIFIED",
                        tables=[contract.table],
                        columns=[index.column],
                        blocking=policy.inverted_index_blocking,
                        source="FORMAL_ASSET_SQL_AST",
                        expected_evidence=expected,
                        evidence=evidence,
                        gap_codes=[item.code for item in local_gaps],
                        capability={"indexName": index.name, "indexType": index.index_type},
                    )
                )
                gaps.extend(local_gaps)
        return obligations, gaps


def assess_doris_physical_plan(
    *,
    sql: str,
    formal_assets: Sequence[Mapping[str, Any] | APIModel],
    explain_payload: Any,
    partition_requirements: Sequence[PartitionPruningRequirement] = (),
    policy: PhysicalPlanGovernancePolicy | None = None,
) -> PhysicalPlanAssessment:
    return DorisPhysicalPlanGovernor().assess(
        sql=sql,
        formal_assets=formal_assets,
        explain_payload=explain_payload,
        partition_requirements=partition_requirements,
        policy=policy,
    )


def _parse_single_read_query(sql: str) -> exp.Expression:
    if not sql:
        raise ValueError("SQL is required for physical-plan assessment")
    normalized = _normalize_placeholders(sql)
    parsed: list[exp.Expression] = []
    errors: list[Exception] = []
    for dialect in ("doris", "mysql"):
        try:
            parsed = [
                item
                for item in sqlglot.parse(normalized, read=dialect)
                if item is not None
            ]
            break
        except (sqlglot.errors.ParseError, sqlglot.errors.TokenError, ValueError) as exc:
            errors.append(exc)
    if len(parsed) != 1:
        raise ValueError("physical-plan assessment requires exactly one SQL statement")
    expression = parsed[0]
    prohibited = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
    )
    if isinstance(expression, prohibited) or any(
        expression.find(node_type) is not None for node_type in prohibited
    ):
        raise ValueError("physical-plan assessment accepts read-only SQL only")
    if expression.find(exp.Select) is None:
        raise ValueError("physical-plan assessment requires a query expression")
    return expression


def _normalize_placeholders(sql: str) -> str:
    output: list[str] = []
    index = 0
    while index < len(sql):
        if sql[index : index + 2] == "%s":
            output.append("?")
            index += 2
            continue
        output.append(sql[index])
        index += 1
    return "".join(output)


def _referenced_base_tables(expression: exp.Expression) -> tuple[str, ...]:
    cte_names = {
        _key(cte.alias_or_name)
        for cte in expression.find_all(exp.CTE)
        if cte.alias_or_name
    }
    tables: list[str] = []
    for table in expression.find_all(exp.Table):
        name = str(table.name or "").strip()
        if not name or _key(name) in cte_names:
            continue
        key = _key(name)
        if key and key not in tables:
            tables.append(key)
    return tuple(tables)


def _formal_asset_contracts(
    assets: Sequence[Mapping[str, Any] | APIModel],
    referenced_tables: Sequence[str],
) -> tuple[dict[str, DorisPhysicalTableContract], list[PhysicalPlanGap]]:
    candidates: dict[str, list[DorisPhysicalTableContract]] = {}
    malformed: dict[str, str] = {}
    for raw in assets:
        payload = _as_mapping(raw)
        metadata = _nested_metadata(payload)
        table = str(
            _mapping_value(payload, "tableName", "table", "key")
            or _mapping_value(metadata, "tableName", "table", "key")
            or ""
        ).strip()
        if not table:
            continue
        table_key = _key(table)
        status = str(
            _mapping_value(payload, "status", "assetStatus")
            or _mapping_value(metadata, "status", "assetStatus")
            or ""
        ).strip().upper()
        physical = _mapping_value(payload, "physicalMetadata")
        if not isinstance(physical, Mapping):
            physical = _mapping_value(metadata, "physicalMetadata")
        if not isinstance(physical, Mapping):
            physical = {}
        if status != "PUBLISHED":
            malformed[table_key] = "PHYSICAL_ASSET_NOT_PUBLISHED"
            continue
        if not physical:
            malformed[table_key] = "PHYSICAL_METADATA_MISSING"
            continue
        partition_columns = _string_list(
            _mapping_value(physical, "partitionColumns")
        )
        bucket_columns = _string_list(_mapping_value(physical, "bucketColumns"))
        indexes = _inverted_indexes(physical)
        physical_payload = {
            "partitionColumns": partition_columns,
            "partitionCount": _positive_int(_mapping_value(physical, "partitionCount")),
            "bucketColumns": bucket_columns,
            "bucketCount": _positive_int(_mapping_value(physical, "bucketCount")),
            "colocateGroup": str(_mapping_value(physical, "colocateGroup") or "").strip(),
            "invertedIndexes": [item.model_dump(by_alias=True) for item in indexes],
            "ddlHash": str(_mapping_value(physical, "ddlHash") or ""),
            "capturedAt": str(_mapping_value(physical, "capturedAt") or ""),
        }
        contract = DorisPhysicalTableContract(
            table=table,
            topic=str(
                _mapping_value(payload, "topic")
                or _mapping_value(metadata, "topic")
                or ""
            ).strip(),
            asset_ref=str(
                _mapping_value(payload, "groundedEvidenceRef", "sourceRefId", "assetRef")
                or _mapping_value(metadata, "groundedEvidenceRef", "sourceRefId", "assetRef")
                or ""
            ).strip(),
            asset_status=status,
            partition_columns=partition_columns,
            partition_count=physical_payload["partitionCount"],
            bucket_columns=bucket_columns,
            bucket_count=physical_payload["bucketCount"],
            colocate_group=physical_payload["colocateGroup"],
            inverted_indexes=indexes,
            metadata_fingerprint=_fingerprint_json(physical_payload),
        )
        candidates.setdefault(table_key, []).append(contract)

    contracts: dict[str, DorisPhysicalTableContract] = {}
    gaps: list[PhysicalPlanGap] = []
    for table_key in referenced_tables:
        matches = candidates.get(table_key, [])
        if len(matches) > 1:
            gaps.append(
                PhysicalPlanGap(
                    code="PHYSICAL_ASSET_AMBIGUOUS",
                    message="More than one published formal physical asset matches a referenced table",
                    tables=[item.table for item in matches],
                    blocking=True,
                    expected_evidence="one active published physical asset per referenced table",
                )
            )
            continue
        if matches:
            contracts[table_key] = matches[0]
            continue
        code = malformed.get(table_key, "FORMAL_PHYSICAL_ASSET_MISSING")
        message = {
            "PHYSICAL_ASSET_NOT_PUBLISHED": "The referenced table's physical asset is not PUBLISHED",
            "PHYSICAL_METADATA_MISSING": "The referenced published asset has no physicalMetadata",
        }.get(code, "No published formal physical asset matches a referenced table")
        gaps.append(
            PhysicalPlanGap(
                code=code,
                message=message,
                tables=[table_key],
                blocking=True,
                expected_evidence="published formal asset with physicalMetadata",
            )
        )
    return contracts, gaps


def _inspect_sql(
    expression: exp.Expression,
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> _ParsedSql:
    filter_columns: set[tuple[str, str]] = set()
    equality_filters: set[tuple[str, str]] = set()
    join_pairs: set[tuple[tuple[str, str], tuple[str, str]]] = set()
    lineage_gaps: list[PhysicalPlanGap] = []
    states: dict[int, _ScopedSqlState] = {}
    has_join = False
    try:
        scopes = list(traverse_scope(expression))
    except Exception as exc:
        lineage_gaps.append(
            PhysicalPlanGap(
                code="PHYSICAL_SQL_SCOPE_ANALYSIS_FAILED",
                message="SQL scopes could not be resolved for physical-plan evidence",
                blocking=True,
                expected_evidence="scope-resolved SQL AST lineage",
                details={"scopeError": str(exc)[:500]},
            )
        )
        scopes = []

    for scope_index, scope in enumerate(scopes):
        state = _build_scoped_sql_state(scope, states, contracts)
        select = scope.expression
        if isinstance(select, exp.Select):
            filter_roots = [
                wrapper.this
                for name in ("where", "having", "qualify")
                for wrapper in [select.args.get(name)]
                if isinstance(wrapper, (exp.Where, exp.Having, exp.Qualify))
                and isinstance(wrapper.this, exp.Expression)
            ]
            for root in filter_roots:
                for column in _direct_nodes(root, exp.Column):
                    origins = _resolve_scoped_column(column, state, contracts)
                    if len(origins) == 1:
                        filter_columns.add(next(iter(origins)))
                    elif _physical_filter_lineage_relevant(
                        column,
                        state,
                        contracts,
                    ):
                        _append_lineage_gap(
                            lineage_gaps,
                            code="PHYSICAL_FILTER_LINEAGE_UNRESOLVED",
                            message="A physical filter column cannot be bound to exactly one base-table scan",
                            expression=root,
                            column=column,
                            state=state,
                            scope_index=scope_index,
                        )
                for equality in _direct_nodes(root, exp.EQ):
                    _collect_scoped_equality(
                        equality,
                        state,
                        contracts,
                        equality_filters,
                        join_pairs,
                    )
                for membership in _direct_nodes(root, exp.In):
                    target = membership.this
                    if not isinstance(target, exp.Column):
                        continue
                    if membership.args.get("query") is not None:
                        continue
                    if any(
                        _direct_contains_column(item)
                        for item in membership.expressions
                    ):
                        continue
                    origins = _resolve_scoped_column(target, state, contracts)
                    if len(origins) == 1:
                        equality_filters.add(next(iter(origins)))

            joins = [
                item
                for item in (select.args.get("joins") or [])
                if isinstance(item, exp.Join)
            ]
            has_join = has_join or bool(joins)
            for join in joins:
                predicate = join.args.get("on")
                if not isinstance(predicate, exp.Expression):
                    if join.args.get("using") and _scope_has_bucket_capability(
                        state,
                        contracts,
                    ):
                        _append_lineage_gap(
                            lineage_gaps,
                            code="PHYSICAL_JOIN_LINEAGE_UNRESOLVED",
                            message="A physical join cannot be bound to explicit base-table equality lineage",
                            expression=join,
                            column=None,
                            state=state,
                            scope_index=scope_index,
                        )
                    continue
                for column in _direct_nodes(predicate, exp.Column):
                    origins = _resolve_scoped_column(column, state, contracts)
                    if len(origins) != 1 and _scope_has_bucket_capability(
                        state,
                        contracts,
                    ):
                        _append_lineage_gap(
                            lineage_gaps,
                            code="PHYSICAL_JOIN_LINEAGE_UNRESOLVED",
                            message="A physical join column cannot be bound to exactly one base-table scan",
                            expression=predicate,
                            column=column,
                            state=state,
                            scope_index=scope_index,
                        )
                for equality in _direct_nodes(predicate, exp.EQ):
                    _collect_scoped_equality(
                        equality,
                        state,
                        contracts,
                        equality_filters,
                        join_pairs,
                    )

        state = _state_with_scope_outputs(scope, state, states, contracts)
        states[id(scope)] = state

    return _ParsedSql(
        expression=expression,
        referenced_tables=tuple(contracts),
        filter_columns=frozenset(filter_columns),
        equality_filter_columns=frozenset(equality_filters),
        join_pairs=frozenset(join_pairs),
        has_join=has_join,
        lineage_gaps=tuple(lineage_gaps),
    )


def _build_scoped_sql_state(
    scope: Scope,
    states: Mapping[int, _ScopedSqlState],
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> _ScopedSqlState:
    sources: dict[str, _ScopedSource] = {}
    for raw_alias, pair in (scope.selected_sources or {}).items():
        alias = _key(raw_alias)
        source = pair[1]
        if isinstance(source, exp.Table):
            table = _key(source.name)
            if table in contracts:
                sources[alias] = _ScopedSource(
                    alias=alias,
                    base_table=table,
                    base_tables=frozenset({table}),
                )
            continue
        if isinstance(source, Scope):
            child = states.get(id(source))
            if child is None:
                sources[alias] = _ScopedSource(alias=alias)
            else:
                sources[alias] = _ScopedSource(
                    alias=alias,
                    outputs=child.outputs,
                    base_tables=child.base_tables,
                )
    base_tables = frozenset(
        table
        for source in sources.values()
        for table in source.base_tables
    )
    return _ScopedSqlState(
        sources=sources,
        outputs={},
        output_order=(),
        base_tables=base_tables,
    )


def _state_with_scope_outputs(
    scope: Scope,
    state: _ScopedSqlState,
    states: Mapping[int, _ScopedSqlState],
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> _ScopedSqlState:
    outputs: dict[str, frozenset[tuple[str, str]]] = {}
    output_order: list[str] = []
    if isinstance(scope.expression, exp.Select):
        for projection in scope.expression.selects:
            projected = projection.this if isinstance(projection, exp.Alias) else projection
            if isinstance(projected, exp.Star):
                _merge_star_outputs(outputs, state.sources.values(), contracts)
                for name in outputs:
                    if name not in output_order:
                        output_order.append(name)
                continue
            if isinstance(projected, exp.Column) and isinstance(projected.this, exp.Star):
                source = state.sources.get(_key(projected.table))
                if source is not None:
                    _merge_star_outputs(outputs, [source], contracts)
                    for name in outputs:
                        if name not in output_order:
                            output_order.append(name)
                continue
            name = _key(projection.alias_or_name)
            if not name:
                continue
            origins: set[tuple[str, str]] = set()
            for column in _direct_nodes(projected, exp.Column):
                origins.update(_resolve_scoped_column(column, state, contracts))
            outputs[name] = frozenset(origins)
            output_order.append(name)
    elif isinstance(scope.expression, (exp.Union, exp.Intersect, exp.Except)):
        children = [
            states.get(id(child))
            for child in (scope.union_scopes or [])
        ]
        child_states = [child for child in children if child is not None]
        if child_states:
            output_order = list(child_states[0].output_order)
            for index, name in enumerate(output_order):
                origins: set[tuple[str, str]] = set()
                for child in child_states:
                    if index >= len(child.output_order):
                        continue
                    child_name = child.output_order[index]
                    origins.update(child.outputs.get(child_name, frozenset()))
                outputs[name] = frozenset(origins)
    base_tables = set(state.base_tables)
    for child in (scope.union_scopes or []):
        child_state = states.get(id(child))
        if child_state is not None:
            base_tables.update(child_state.base_tables)
    return _ScopedSqlState(
        sources=state.sources,
        outputs=outputs,
        output_order=tuple(output_order),
        base_tables=frozenset(base_tables),
    )


def _merge_star_outputs(
    outputs: dict[str, frozenset[tuple[str, str]]],
    sources: Iterable[_ScopedSource],
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> None:
    for source in sources:
        source_outputs: dict[str, frozenset[tuple[str, str]]] = {}
        if source.base_table:
            contract = contracts.get(source.base_table)
            if contract is not None:
                source_outputs = {
                    column: frozenset({(source.base_table, column)})
                    for column in _declared_physical_columns(contract)
                }
        elif source.outputs:
            source_outputs = dict(source.outputs)
        for name, origins in source_outputs.items():
            outputs[name] = frozenset(
                {*outputs.get(name, frozenset()), *origins}
            )


def _resolve_scoped_column(
    column: exp.Column,
    state: _ScopedSqlState,
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> frozenset[tuple[str, str]]:
    column_key = _key(column.name)
    if not column_key:
        return frozenset()
    qualifier = _key(column.table)
    if qualifier:
        source = state.sources.get(qualifier)
        return _source_column_origins(source, column_key)
    if len(state.sources) == 1:
        source = next(iter(state.sources.values()))
        return _source_column_origins(source, column_key)
    origins: set[tuple[str, str]] = set()
    for source in state.sources.values():
        if source.base_table:
            contract = contracts.get(source.base_table)
            if contract is None or column_key not in _declared_physical_columns(contract):
                continue
            origins.add((source.base_table, column_key))
            continue
        if source.outputs and column_key in source.outputs:
            origins.update(source.outputs[column_key])
    return frozenset(origins)


def _source_column_origins(
    source: _ScopedSource | None,
    column_key: str,
) -> frozenset[tuple[str, str]]:
    if source is None:
        return frozenset()
    if source.base_table:
        return frozenset({(source.base_table, column_key)})
    if source.outputs and column_key in source.outputs:
        return source.outputs[column_key]
    return frozenset()


def _collect_scoped_equality(
    equality: exp.EQ,
    state: _ScopedSqlState,
    contracts: Mapping[str, DorisPhysicalTableContract],
    equality_filters: set[tuple[str, str]],
    join_pairs: set[tuple[tuple[str, str], tuple[str, str]]],
) -> None:
    left_expression = equality.this
    right_expression = equality.expression
    left_origins = (
        _resolve_scoped_column(left_expression, state, contracts)
        if isinstance(left_expression, exp.Column)
        else frozenset()
    )
    right_origins = (
        _resolve_scoped_column(right_expression, state, contracts)
        if isinstance(right_expression, exp.Column)
        else frozenset()
    )
    if len(left_origins) == 1 and len(right_origins) == 1:
        left = next(iter(left_origins))
        right = next(iter(right_origins))
        if left != right:
            join_pairs.add(_ordered_join_pair((left, right)))
        return
    if len(left_origins) == 1 and not _direct_contains_column(right_expression):
        equality_filters.add(next(iter(left_origins)))
    if len(right_origins) == 1 and not _direct_contains_column(left_expression):
        equality_filters.add(next(iter(right_origins)))


def _direct_contains_column(expression: Any) -> bool:
    return bool(
        isinstance(expression, exp.Expression)
        and _direct_nodes(expression, exp.Column)
    )


def _direct_nodes(
    expression: exp.Expression,
    node_type: type[exp.Expression],
) -> list[Any]:
    found: list[Any] = []

    def visit(node: Any, *, root: bool = False) -> None:
        if not isinstance(node, exp.Expression):
            return
        if not root and isinstance(node, (exp.Select, exp.Subquery)):
            return
        if isinstance(node, node_type):
            found.append(node)
        for child in node.iter_expressions():
            visit(child)

    visit(expression, root=True)
    return found


def _physical_filter_lineage_relevant(
    column: exp.Column,
    state: _ScopedSqlState,
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> bool:
    column_key = _key(column.name)
    if not column_key:
        return False
    return any(
        table in contracts
        and column_key in _declared_physical_columns(contracts[table])
        for table in state.base_tables
    )


def _scope_has_bucket_capability(
    state: _ScopedSqlState,
    contracts: Mapping[str, DorisPhysicalTableContract],
) -> bool:
    return any(
        table in contracts and bool(contracts[table].bucket_columns)
        for table in state.base_tables
    )


def _append_lineage_gap(
    gaps: list[PhysicalPlanGap],
    *,
    code: str,
    message: str,
    expression: exp.Expression,
    column: exp.Column | None,
    state: _ScopedSqlState,
    scope_index: int,
) -> None:
    tables = sorted(state.base_tables)
    columns = [str(column.name)] if column is not None and column.name else []
    rendered = expression.sql(dialect="doris")[:500]
    identity = (code, tuple(tables), tuple(_key(item) for item in columns), rendered)
    if any(
        (
            item.code,
            tuple(sorted(_key(table) for table in item.tables)),
            tuple(_key(item_column) for item_column in item.columns),
            str(item.details.get("expression") or ""),
        )
        == identity
        for item in gaps
    ):
        return
    gaps.append(
        PhysicalPlanGap(
            code=code,
            message=message,
            kind="SQL_AST_LINEAGE",
            tables=tables,
            columns=columns,
            blocking=True,
            expected_evidence="one scope-resolved base-table origin per physical SQL column",
            details={"scopeIndex": scope_index, "expression": rendered},
        )
    )


def _declared_physical_columns(contract: DorisPhysicalTableContract) -> set[str]:
    return {
        _key(item)
        for item in [
            *contract.partition_columns,
            *contract.bucket_columns,
            *[index.column for index in contract.inverted_indexes],
        ]
        if item
    }


def _expected_bucket_join_pairs(
    left: DorisPhysicalTableContract,
    right: DorisPhysicalTableContract,
) -> set[tuple[tuple[str, str], tuple[str, str]]]:
    if len(left.bucket_columns) != len(right.bucket_columns):
        return set()
    return {
        _ordered_join_pair(
            (
                (_key(left.table), _key(left_column)),
                (_key(right.table), _key(right_column)),
            )
        )
        for left_column, right_column in zip(
            left.bucket_columns,
            right.bucket_columns,
        )
    }


def _ordered_join_pair(
    pair: tuple[tuple[str, str], tuple[str, str]],
) -> tuple[tuple[str, str], tuple[str, str]]:
    return tuple(sorted(pair))  # type: ignore[return-value]


def _explain_lines(
    payload: Any,
    referenced_tables: Sequence[str],
) -> list[_ExplainLine]:
    raw_lines: list[tuple[str, str, str]] = []
    _flatten_explain_payload(payload, "root", "", raw_lines)
    known_tables = tuple(referenced_tables)
    active_tables: tuple[str, ...] = ()
    result: list[_ExplainLine] = []
    for locator, field, text in raw_lines:
        tokens = tuple(_tokens(text))
        direct = tuple(
            table for table in known_tables if _identifier_in_tokens(table, tokens)
        )
        if direct:
            active_tables = direct
        result.append(
            _ExplainLine(
                locator=locator,
                field=field,
                text=text,
                tokens=tokens,
                direct_table_refs=direct,
                table_refs=direct or active_tables,
            )
        )
    return result


def _flatten_explain_payload(
    payload: Any,
    locator: str,
    field: str,
    output: list[tuple[str, str, str]],
) -> None:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            name = str(key or "")
            _flatten_explain_payload(value, "%s.%s" % (locator, name), name, output)
        return
    if isinstance(payload, (list, tuple)):
        for index, value in enumerate(payload):
            _flatten_explain_payload(value, "%s[%d]" % (locator, index), field, output)
        return
    if payload is None:
        return
    text = str(payload)
    for line_number, line in enumerate(text.splitlines() or [text]):
        clean = line.strip()
        if not clean:
            continue
        rendered = "%s: %s" % (field, clean) if field else clean
        output.append(("%s:%d" % (locator, line_number), field, rendered))


def _ratio_plan_evidence(
    lines: Sequence[_ExplainLine],
    table: str,
    *,
    kind: Literal["partition", "tablet"],
    known_total: int,
    referenced_table_count: int,
) -> tuple[PhysicalPlanEvidence | None, str, dict[str, Any]]:
    table_key = _key(table)
    term_prefixes = (
        ("partition", "partitions", "partitionratio")
        if kind == "partition"
        else ("tablet", "tablets", "tabletratio", "bucket", "buckets")
    )
    observed: list[dict[str, Any]] = []
    for line in lines:
        if not _line_applies_to_table(line, table_key, referenced_table_count):
            continue
        field_key = _key(line.field)
        has_term = any(
            token.startswith(prefix)
            for token in (*line.tokens, field_key)
            for prefix in term_prefixes
        )
        if not has_term:
            continue
        for selected, total in _integer_ratios(line.text):
            details = {
                "selected": selected,
                "total": total,
                "knownTotal": known_total,
            }
            observed.append(details)
            if total > 0 and (
                selected < total
                or (selected == total == 1 and known_total == 1)
            ):
                return (
                    PhysicalPlanEvidence(
                        source="EXPLAIN_VERBOSE",
                        locator=line.locator,
                        excerpt=line.text[:500],
                        attributes=details,
                    ),
                    "",
                    details,
                )
    prefix = "PARTITION" if kind == "partition" else "TABLET"
    if observed:
        return None, "%s_SCAN_NOT_REDUCED" % prefix, {"observedRatios": observed}
    return None, "%s_PLAN_EVIDENCE_MISSING" % prefix, {}


def _colocate_plan_evidence(
    lines: Sequence[_ExplainLine],
    left_table: str,
    right_table: str,
) -> PhysicalPlanEvidence | None:
    required_tables = {_key(left_table), _key(right_table)}
    for line in lines:
        if not required_tables.issubset(set(line.table_refs)):
            continue
        has_colocate = any(token.startswith("colocate") for token in line.tokens)
        if not has_colocate:
            continue
        negative = any(
            token in {"false", "disabled", "unused", "none", "no", "not"}
            for token in line.tokens
        )
        positive = any(
            token in {"true", "enabled", "yes"} for token in line.tokens
        )
        if positive and not negative:
            return PhysicalPlanEvidence(
                source="EXPLAIN_VERBOSE",
                locator=line.locator,
                excerpt=line.text[:500],
                attributes={
                    "tables": [left_table, right_table],
                    "tablePairBound": True,
                },
            )
    return None


def _inverted_index_plan_evidence(
    lines: Sequence[_ExplainLine],
    *,
    table: str,
    column: str,
    index_name: str,
    index_name_unique: bool,
    referenced_table_count: int,
) -> PhysicalPlanEvidence | None:
    table_key = _key(table)
    column_key = _key(column)
    index_key = _key(index_name)
    for line in lines:
        table_scoped = _line_applies_to_table(
            line,
            table_key,
            referenced_table_count,
        )
        unique_named = bool(
            index_key
            and index_name_unique
            and _identifier_in_tokens(index_key, line.tokens)
        )
        if not table_scoped and not unique_named:
            continue
        engine_index_term = any(
            token.startswith("inverted") or token.startswith("index")
            for token in line.tokens
        )
        named = bool(index_key and _identifier_in_tokens(index_key, line.tokens))
        column_named = _identifier_in_tokens(column_key, line.tokens)
        negative = any(
            token in {"false", "disabled", "unused", "none", "no", "not"}
            for token in line.tokens
        )
        if engine_index_term and not negative and (named or column_named):
            return PhysicalPlanEvidence(
                source="EXPLAIN_VERBOSE",
                locator=line.locator,
                excerpt=line.text[:500],
                attributes={"indexName": index_name, "column": column},
            )
    return None


def _line_applies_to_table(
    line: _ExplainLine,
    table_key: str,
    referenced_table_count: int,
) -> bool:
    if table_key in line.table_refs:
        return True
    return referenced_table_count == 1 and not line.table_refs


def _integer_ratios(text: str) -> list[tuple[int, int]]:
    ratios: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if not text[index].isdigit():
            index += 1
            continue
        left_start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        left_text = text[left_start:index]
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != "/":
            continue
        index += 1
        while index < len(text) and text[index].isspace():
            index += 1
        right_start = index
        while index < len(text) and text[index].isdigit():
            index += 1
        if right_start == index:
            continue
        ratios.append((int(left_text), int(text[right_start:index])))
    return ratios


def _inverted_indexes(physical: Mapping[str, Any]) -> list[DorisInvertedIndexContract]:
    indexes: list[DorisInvertedIndexContract] = []
    seen: set[tuple[str, str]] = set()
    raw_indexes = _mapping_value(physical, "invertedIndexes")
    if isinstance(raw_indexes, Sequence) and not isinstance(raw_indexes, (str, bytes)):
        for raw in raw_indexes:
            if not isinstance(raw, Mapping):
                continue
            index_type = str(_mapping_value(raw, "type", "indexType") or "INVERTED").strip().upper()
            if index_type != "INVERTED":
                continue
            name = str(_mapping_value(raw, "name", "indexName", "keyName") or "").strip()
            columns = _string_list(_mapping_value(raw, "columns"))
            single = str(_mapping_value(raw, "column", "columnName") or "").strip()
            if single:
                columns.insert(0, single)
            for column in _dedupe_strings(columns):
                identity = (_key(name), _key(column))
                if not column or identity in seen:
                    continue
                seen.add(identity)
                indexes.append(
                    DorisInvertedIndexContract(
                        name=name,
                        column=column,
                        index_type=index_type,
                    )
                )
    for column in _string_list(_mapping_value(physical, "invertedIndexColumns")):
        identity = ("", _key(column))
        if any(existing[1] == identity[1] for existing in seen):
            continue
        seen.add(identity)
        indexes.append(DorisInvertedIndexContract(column=column))
    return indexes


def _asset_evidence(
    contract: DorisPhysicalTableContract,
    attributes: Mapping[str, Any],
) -> PhysicalPlanEvidence:
    return PhysicalPlanEvidence(
        source="FORMAL_ASSET",
        locator=contract.asset_ref or contract.metadata_fingerprint,
        attributes={
            "table": contract.table,
            "metadataFingerprint": contract.metadata_fingerprint,
            **dict(attributes),
        },
    )


def _gap(
    *,
    code: str,
    message: str,
    obligation_id: str,
    kind: str,
    tables: Sequence[str],
    columns: Sequence[str],
    blocking: bool,
    expected: str,
    details: Mapping[str, Any] | None = None,
) -> PhysicalPlanGap:
    return PhysicalPlanGap(
        code=code,
        message=message,
        obligation_id=obligation_id,
        kind=kind,
        tables=list(tables),
        columns=list(columns),
        blocking=blocking,
        expected_evidence=expected,
        details=dict(details or {}),
    )


def _final_assessment(
    *,
    base: Mapping[str, Any],
    gaps: Sequence[PhysicalPlanGap],
    invalid: bool = False,
) -> PhysicalPlanAssessment:
    payload = dict(base)
    obligations = list(payload.get("obligations") or [])
    if invalid:
        status = "INVALID"
    elif gaps:
        status = "GAPPED"
    elif obligations:
        status = "VERIFIED"
    else:
        status = "NOT_REQUIRED"
    payload["status"] = status
    payload["gaps"] = list(gaps)
    payload["executable"] = not any(item.blocking for item in gaps)
    identity = {
        "version": "doris_physical_plan_assessment.v1",
        "sqlFingerprint": payload.get("sql_fingerprint", ""),
        "explainFingerprint": payload.get("explain_fingerprint", ""),
        "policyFingerprint": payload.get("policy_fingerprint", ""),
        "assets": [
            item.metadata_fingerprint
            for item in payload.get("formal_assets", [])
            if isinstance(item, DorisPhysicalTableContract)
        ],
        "obligations": [
            item.obligation_id
            for item in obligations
            if isinstance(item, PhysicalPlanObligationAssessment)
        ],
        "gaps": [item.code for item in gaps],
    }
    payload["assessment_id"] = _fingerprint_json(identity)
    return PhysicalPlanAssessment(**payload)


def _obligation_id(
    kind: str,
    tables: Sequence[str],
    columns: Sequence[str],
    evidence_ref: str = "",
) -> str:
    return _fingerprint_json(
        {
            "kind": kind,
            "tables": [_key(item) for item in tables],
            "columns": [_key(item) for item in columns],
            "evidenceRef": str(evidence_ref or "").strip(),
        }
    )[:24]


def _fingerprint_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _fingerprint_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_mapping(value: Mapping[str, Any] | APIModel) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return value.model_dump(by_alias=True, mode="python")


def _nested_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    value = _mapping_value(payload, "metadata")
    return value if isinstance(value, Mapping) else {}


def _mapping_value(payload: Mapping[str, Any], *names: str) -> Any:
    normalized = {_key(name): value for name, value in payload.items()}
    for name in names:
        if _key(name) in normalized:
            return normalized[_key(name)]
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return _dedupe_strings([str(item or "").strip() for item in value])


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        identity = _key(text)
        if not text or identity in seen:
            continue
        seen.add(identity)
        result.append(text)
    return result


def _positive_int(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return max(0, parsed)


def _key(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "").strip()
        if character.isalnum() or character == "_"
    )


def _tokens(value: Any) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for character in str(value or "").casefold():
        if character.isalnum() or character == "_":
            current.append(character)
            continue
        if current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def _identifier_in_tokens(identifier: str, tokens: Sequence[str]) -> bool:
    target = _key(identifier)
    return bool(target and target in tokens)
