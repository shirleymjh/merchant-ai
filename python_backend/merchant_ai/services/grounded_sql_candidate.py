from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Iterable, Mapping, Sequence

import sqlglot
from pydantic import Field
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_query_contract import (
    GroundedQueryContract,
    GroundedRelationshipBinding,
)


class GroundedSqlCandidate(APIModel):
    """A complete SQL proposal authored by Core after semantic grounding.

    The candidate deliberately carries SQL rather than a business-shaped query
    template.  Its optional contract fingerprint prevents a proposal produced
    for one grounded contract from being replayed against another one.
    """

    candidate_version: str = "grounded_sql_candidate.v1"
    sql: str
    contract_fingerprint: str = ""
    evidence_refs: list[str] = Field(default_factory=list)
    rationale: str = ""


class GroundedSqlValidationGap(APIModel):
    code: str
    message: str
    blocking: bool = True
    table: str = ""
    column: str = ""
    relationship_ref_id: str = ""
    resolution: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class GroundedSqlValidationResult(APIModel):
    valid: bool = False
    canonical_sql: str = ""
    ast_fingerprint: str = ""
    contract_fingerprint: str = ""
    referenced_tables: list[str] = Field(default_factory=list)
    referenced_columns: list[str] = Field(default_factory=list)
    relationship_refs: list[str] = Field(default_factory=list)
    output_columns: list[str] = Field(default_factory=list)
    output_lineage: dict[str, list[str]] = Field(default_factory=dict)
    gaps: list[GroundedSqlValidationGap] = Field(default_factory=list)


@dataclass(frozen=True)
class _Origin:
    table: str
    column: str
    scan_ids: frozenset[str] = frozenset()


@dataclass
class _Output:
    origins: set[_Origin] = field(default_factory=set)
    passthrough: bool = False
    semantic_refs: set[str] = field(default_factory=set)
    exact_metric_refs: set[str] = field(default_factory=set)


@dataclass
class _Source:
    alias: str
    base_table: str = ""
    scan_id: str = ""
    outputs: dict[str, _Output] = field(default_factory=dict)
    base_tables: set[str] = field(default_factory=set)
    scan_ids: set[str] = field(default_factory=set)

    @property
    def derived(self) -> bool:
        return not bool(self.base_table)


@dataclass
class _ScopeState:
    scope: Any
    sources: dict[str, _Source]
    outputs: dict[str, _Output] = field(default_factory=dict)
    base_tables: set[str] = field(default_factory=set)
    scan_ids: set[str] = field(default_factory=set)


@dataclass
class _ResolvedColumn:
    status: str
    origins: set[_Origin] = field(default_factory=set)
    source_alias: str = ""
    derived_output: bool = False


@dataclass(frozen=True)
class _JoinProof:
    relationship_ref_id: str
    relationship_name: str
    relationship: GroundedRelationshipBinding
    forward: bool
    left_scan_ids: frozenset[str]
    right_scan_ids: frozenset[str]
    left_tables: frozenset[str]
    right_tables: frozenset[str]


@dataclass(frozen=True)
class _MetricFormulaObligation:
    metric_index: int
    semantic_ref_id: str
    metric_key: str
    table: str
    source_columns: frozenset[str]
    signature: str
    parse_error: str = ""


@dataclass(frozen=True)
class _TimeObligation:
    obligation_index: int
    table: str
    column: str
    start_value: str = ""
    end_value: str = ""


@dataclass
class _ValidationContext:
    contract: GroundedQueryContract
    trusted_refs: set[str]
    allowed_tables: dict[str, str]
    allowed_columns: dict[str, set[str]]
    merchant_columns: dict[str, str]
    relationships: list[GroundedRelationshipBinding]
    scope_states: dict[int, _ScopeState] = field(default_factory=dict)
    gaps: list[GroundedSqlValidationGap] = field(default_factory=list)
    referenced_tables: set[str] = field(default_factory=set)
    referenced_columns: set[str] = field(default_factory=set)
    relationship_refs: set[str] = field(default_factory=set)
    entity_predicate_coverage: dict[int, set[str]] = field(default_factory=dict)
    entity_scans: dict[int, set[str]] = field(default_factory=dict)
    metric_formula_obligations: list[_MetricFormulaObligation] = field(default_factory=list)
    metric_formula_coverage: set[int] = field(default_factory=set)
    metric_formula_scopes: dict[int, set[int]] = field(default_factory=dict)
    time_obligations: list[_TimeObligation] = field(default_factory=list)
    time_scans: dict[int, set[str]] = field(default_factory=dict)
    time_lower_coverage: dict[int, set[str]] = field(default_factory=dict)
    time_upper_coverage: dict[int, set[str]] = field(default_factory=dict)
    column_binding_refs: dict[tuple[str, str], set[str]] = field(default_factory=dict)
    final_outputs: dict[str, _Output] = field(default_factory=dict)
    final_output_columns: list[str] = field(default_factory=list)
    join_proofs: list[tuple[_JoinProof, _ScopeState]] = field(default_factory=list)


class GroundedSqlCandidateValidator:
    """Validate arbitrary read-only SQL against one grounded semantic contract.

    This validator contains no question, Topic, table-name, or metric-specific
    branches.  Core remains free to author CTEs, windows, aggregates, nested
    selects and an arbitrary number of governed joins.  The validator only
    proves that every physical object and relationship came from semantic
    evidence already bound into the contract.
    """

    dialect = "doris"

    def validate(
        self,
        candidate: GroundedSqlCandidate | Mapping[str, Any] | str,
        contract: GroundedQueryContract,
    ) -> GroundedSqlValidationResult:
        proposed = _normalize_candidate(candidate)
        contract_fp = grounded_query_contract_fingerprint(contract)
        preliminary: list[GroundedSqlValidationGap] = []
        if proposed.contract_fingerprint and proposed.contract_fingerprint != contract_fp:
            preliminary.append(
                _gap(
                    "SQL_CONTRACT_FINGERPRINT_MISMATCH",
                    "SQL candidate was authored for a different grounded contract",
                    resolution="Regenerate SQL from the current GroundedQueryContract.",
                )
            )
        if not bool(getattr(contract, "ready", False)):
            preliminary.append(
                _gap(
                    "GROUNDED_CONTRACT_NOT_READY",
                    "Only a READY grounded contract may authorize LLM-authored SQL",
                    resolution="Resolve the contract's blocking semantic gaps first.",
                )
            )

        parsed, parse_gaps = self._parse_single_read_only(proposed.sql)
        preliminary.extend(parse_gaps)
        if parsed is None:
            return GroundedSqlValidationResult(
                valid=False,
                contract_fingerprint=contract_fp,
                gaps=_dedupe_gaps(preliminary),
            )
        if any(isinstance(node, exp.SetOperation) for node in parsed.walk()):
            preliminary.append(
                _gap(
                    "SQL_SET_OPERATION_UNPROVEN",
                    "UNION/INTERSECT/EXCEPT are blocked until every branch can carry an independent grounded proof",
                    resolution="Use one grounded SELECT, or bind and validate every set-operation branch separately.",
                )
            )

        canonical = _canonical_sql(parsed, self.dialect)
        ast_fingerprint = _ast_fingerprint(canonical, self.dialect)
        context = _build_context(contract)
        context.gaps.extend(preliminary)
        self._validate_scopes(parsed, context)
        self._validate_metric_formula_obligations(context)
        self._validate_metric_fanout(context)
        self._validate_entity_obligations(context)
        self._validate_time_obligations(context)
        self._validate_final_outputs(context)

        gaps = _dedupe_gaps(context.gaps)
        return GroundedSqlValidationResult(
            valid=not any(item.blocking for item in gaps),
            canonical_sql=canonical,
            ast_fingerprint=ast_fingerprint,
            contract_fingerprint=contract_fp,
            referenced_tables=sorted(context.referenced_tables),
            referenced_columns=sorted(context.referenced_columns),
            relationship_refs=sorted(context.relationship_refs),
            output_columns=list(context.final_output_columns),
            output_lineage={
                alias: sorted(
                    "%s.%s" % (origin.table, origin.column)
                    for origin in output.origins
                )
                for alias, output in context.final_outputs.items()
            },
            gaps=gaps,
        )

    def _parse_single_read_only(
        self,
        sql: str,
    ) -> tuple[exp.Expression | None, list[GroundedSqlValidationGap]]:
        source = str(sql or "").strip()
        if not source:
            return None, [_gap("SQL_EMPTY", "SQL candidate is empty")]
        try:
            statements = [item for item in sqlglot.parse(source, read=self.dialect) if item is not None]
        except Exception as exc:
            return None, [
                _gap(
                    "SQL_PARSE_ERROR",
                    "SQL candidate could not be parsed as Doris SQL",
                    details={"parserError": str(exc)[:500]},
                )
            ]
        if len(statements) != 1:
            return None, [
                _gap(
                    "SQL_SINGLE_STATEMENT_REQUIRED",
                    "SQL candidate must contain exactly one statement",
                    details={"statementCount": len(statements)},
                )
            ]
        parsed = statements[0]
        if not isinstance(parsed, exp.Query):
            return None, [
                _gap(
                    "SQL_READ_ONLY_REQUIRED",
                    "SQL candidate must be a read-only query",
                    details={"statementType": type(parsed).__name__},
                )
            ]
        forbidden_names = {
            "Alter",
            "Analyze",
            "Command",
            "Copy",
            "Create",
            "Delete",
            "Drop",
            "Grant",
            "Insert",
            "Into",
            "LoadData",
            "Lock",
            "Merge",
            "Pragma",
            "Revoke",
            "Set",
            "Transaction",
            "TruncateTable",
            "Unload",
            "Update",
            "Use",
        }
        forbidden = next(
            (node for node in parsed.walk() if type(node).__name__ in forbidden_names),
            None,
        )
        if forbidden is not None:
            return None, [
                _gap(
                    "SQL_READ_ONLY_REQUIRED",
                    "SQL candidate contains a non-read-only operation",
                    details={"operation": type(forbidden).__name__},
                )
            ]
        return parsed, []

    def _validate_scopes(self, parsed: exp.Expression, context: _ValidationContext) -> None:
        try:
            scopes = list(traverse_scope(parsed))
        except Exception as exc:
            context.gaps.append(
                _gap(
                    "SQL_SCOPE_ANALYSIS_FAILED",
                    "SQL aliases and scopes could not be resolved safely",
                    details={"scopeError": str(exc)[:500]},
                )
            )
            return
        for scope_index, scope in enumerate(scopes):
            state = self._build_scope_state(scope, scope_index, context)
            context.scope_states[id(scope)] = state
            self._validate_scope_columns(state, context)
            self._validate_projection_wildcards(state, context)
            self._validate_correlated_scope(state, context)
            self._validate_scope_joins(state, context)
            self._validate_tenant_predicates(state, context)
            self._record_metric_formulas(state, context)
            self._record_entity_predicates(state, context)
            self._record_time_predicates(state, context)
            state.outputs = self._build_scope_outputs(state, context)
            state.base_tables = set().union(
                *(source.base_tables for source in state.sources.values())
            ) if state.sources else set()
            state.scan_ids = set().union(
                *(source.scan_ids for source in state.sources.values())
            ) if state.sources else set()
            if getattr(scope, "parent", None) is None:
                context.final_outputs = dict(state.outputs)
                context.final_output_columns = _select_output_names(scope.expression)

    def _build_scope_state(
        self,
        scope: Any,
        scope_index: int,
        context: _ValidationContext,
    ) -> _ScopeState:
        sources: dict[str, _Source] = {}
        for raw_alias, pair in (getattr(scope, "selected_sources", {}) or {}).items():
            alias = _identifier(raw_alias)
            source = pair[1]
            if isinstance(source, exp.Table):
                table = _identifier(source.name)
                declared = context.allowed_tables.get(table, "")
                if not declared:
                    context.gaps.append(
                        _gap(
                            "SQL_TABLE_NOT_GROUNDED",
                            "Table %s is not a trusted table binding in the grounded contract" % source.sql(),
                            table=source.sql(),
                            resolution="Read and bind the table detail before using it in SQL.",
                        )
                    )
                else:
                    qualifier = _table_qualifier(source)
                    if qualifier and _identifier(declared) == table and "." not in declared:
                        context.gaps.append(
                            _gap(
                                "SQL_TABLE_QUALIFIER_NOT_GROUNDED",
                                "Qualified table %s is broader than the grounded table identity" % source.sql(),
                                table=source.sql(),
                            )
                        )
                    context.referenced_tables.add(declared)
                scan_id = "%d:%s:%s" % (scope_index, alias, table)
                sources[alias] = _Source(
                    alias=alias,
                    base_table=table,
                    scan_id=scan_id,
                    base_tables={table},
                    scan_ids={scan_id},
                )
                for obligation_index, obligation in enumerate(context.contract.entity_filters):
                    if _identifier(obligation.table) == table:
                        context.entity_scans.setdefault(obligation_index, set()).add(scan_id)
                for obligation in context.time_obligations:
                    if obligation.table == table:
                        context.time_scans.setdefault(obligation.obligation_index, set()).add(scan_id)
                continue
            child_state = context.scope_states.get(id(source))
            if child_state is None:
                context.gaps.append(
                    _gap(
                        "SQL_DERIVED_SOURCE_UNRESOLVED",
                        "Derived source %s could not be grounded to its child query" % alias,
                    )
                )
                sources[alias] = _Source(alias=alias)
                continue
            sources[alias] = _Source(
                alias=alias,
                outputs=dict(child_state.outputs),
                base_tables=set(child_state.base_tables),
                scan_ids=set(child_state.scan_ids),
            )
        return _ScopeState(
            scope=scope,
            sources=sources,
            base_tables=(
                set().union(*(source.base_tables for source in sources.values()))
                if sources
                else set()
            ),
            scan_ids=(
                set().union(*(source.scan_ids for source in sources.values()))
                if sources
                else set()
            ),
        )

    def _validate_scope_columns(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        for column in getattr(state.scope, "columns", []) or []:
            if isinstance(column.this, exp.Star):
                continue
            resolved = _resolve_column(column, state, context.allowed_columns)
            column_name = _identifier(column.name)
            if resolved.status == "unknown":
                context.gaps.append(
                    _gap(
                        "SQL_COLUMN_NOT_GROUNDED",
                        "Column %s does not originate from a trusted contract binding" % column.sql(),
                        column=column.sql(),
                        resolution="Read and bind the field or remove it from the SQL candidate.",
                    )
                )
                continue
            if resolved.status == "ambiguous":
                context.gaps.append(
                    _gap(
                        "SQL_COLUMN_AMBIGUOUS",
                        "Unqualified column %s resolves to more than one SQL source" % column.sql(),
                        column=column.sql(),
                        resolution="Qualify the column with its intended table or CTE alias.",
                    )
                )
                continue
            context.referenced_columns.update(
                "%s.%s" % (origin.table, origin.column)
                for origin in resolved.origins
            )
            if not resolved.origins and not resolved.derived_output and column_name:
                context.gaps.append(
                    _gap(
                        "SQL_DERIVED_COLUMN_UNRESOLVED",
                        "Column %s has no grounded source lineage" % column.sql(),
                        column=column.sql(),
                    )
                )

    def _validate_projection_wildcards(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        expression = state.scope.expression
        if not isinstance(expression, exp.Select):
            return
        for projection in expression.expressions:
            target = projection.this if isinstance(projection, exp.Alias) else projection
            if isinstance(target, exp.Star) or (
                isinstance(target, exp.Column) and isinstance(target.this, exp.Star)
            ):
                context.gaps.append(
                    _gap(
                        "SQL_WILDCARD_NOT_GROUNDED",
                        "SELECT wildcard can expose columns that were not progressively read",
                        column=target.sql(),
                        resolution="Project explicit grounded columns instead of SELECT *.",
                    )
                )

    def _validate_correlated_scope(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        external = [
            column
            for column in (getattr(state.scope, "external_columns", []) or [])
            if _identifier(column.table)
            and _identifier(column.table) not in state.sources
        ]
        if not external:
            return
        # A correlated predicate is a join edge expressed outside JOIN syntax.
        # Until it can carry the same relationship proof, fail closed rather
        # than allowing it to bypass the governed-join validator.
        context.gaps.append(
            _gap(
                "SQL_CORRELATED_RELATIONSHIP_UNPROVEN",
                "Correlated subquery columns require an explicit governed JOIN",
                details={"columns": sorted({item.sql() for item in external})},
                resolution="Express the relationship as JOIN ... ON using a bound relationship edge.",
            )
        )

    def _validate_scope_joins(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            return
        from_clause = select.args.get("from_")
        left_aliases: list[str] = []
        if isinstance(from_clause, exp.From) and from_clause.this is not None:
            alias = _relation_alias(from_clause.this)
            if alias:
                left_aliases.append(alias)
        for join in select.args.get("joins") or []:
            right_alias = _relation_alias(join.this)
            if not right_alias or right_alias not in state.sources:
                context.gaps.append(
                    _gap(
                        "SQL_JOIN_SOURCE_UNRESOLVED",
                        "JOIN source could not be resolved to a grounded SQL source",
                        details={"join": join.sql()},
                    )
                )
                continue
            join_type = _join_type(join)
            on = join.args.get("on")
            using = join.args.get("using") or []
            if join_type == "CROSS" or (on is None and not using):
                context.gaps.append(
                    _gap(
                        "SQL_CARTESIAN_PRODUCT_FORBIDDEN",
                        "Every multi-source query must use an explicit governed relationship predicate",
                        details={"join": join.sql()},
                        resolution="Use JOIN ... ON with every declared relationship key.",
                    )
                )
                left_aliases.append(right_alias)
                continue
            proof = _prove_join(
                state,
                left_aliases,
                right_alias,
                join,
                context,
            )
            if proof is None:
                context.gaps.append(
                    _gap(
                        "SQL_JOIN_NOT_GOVERNED",
                        "JOIN does not match any trusted contract relationship and its complete key set",
                        details={
                            "join": join.sql(),
                            "leftSources": list(left_aliases),
                            "rightSource": right_alias,
                        },
                        resolution="Read and bind the relationship, then use exactly its governed keys.",
                    )
                )
            else:
                context.relationship_refs.add(proof.relationship_ref_id)
                context.join_proofs.append((proof, state))
            left_aliases.append(right_alias)

    def _validate_tenant_predicates(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            return
        predicate_roots: list[tuple[str, exp.Expression]] = []
        for key in ("where", "having", "qualify"):
            wrapper = select.args.get(key)
            if isinstance(wrapper, exp.Expression) and isinstance(wrapper.this, exp.Expression):
                predicate_roots.append((key.upper(), wrapper.this))
        for location, root in predicate_roots:
            for column in _direct_columns(root):
                resolved = _resolve_column(column, state, context.allowed_columns)
                if any(
                    context.merchant_columns.get(origin.table) == origin.column
                    for origin in resolved.origins
                ):
                    context.gaps.append(
                        _gap(
                            "SQL_TENANT_SCOPE_AUTHORED_BY_LLM",
                            "Core SQL may not author tenant predicates; trusted execution injects tenant scope",
                            table=next((origin.table for origin in resolved.origins), ""),
                            column=column.sql(),
                            details={"predicateLocation": location},
                            resolution="Remove the tenant predicate and leave tenant binding to the executor.",
                        )
                    )

        for join in select.args.get("joins") or []:
            on = join.args.get("on")
            if on is None:
                continue
            for term in _and_terms(on):
                tenant_columns: list[exp.Column] = []
                for column in _direct_columns(term):
                    resolved = _resolve_column(column, state, context.allowed_columns)
                    if any(
                        context.merchant_columns.get(origin.table) == origin.column
                        for origin in resolved.origins
                    ):
                        tenant_columns.append(column)
                if not tenant_columns:
                    continue
                if not (
                    isinstance(_unwrap(term), exp.EQ)
                    and isinstance(_unwrap(term).this, exp.Column)
                    and isinstance(_unwrap(term).expression, exp.Column)
                ):
                    context.gaps.append(
                        _gap(
                            "SQL_TENANT_SCOPE_AUTHORED_BY_LLM",
                            "Tenant columns in JOIN may only participate in a governed key equality",
                            column=", ".join(item.sql() for item in tenant_columns),
                        )
                    )

    def _record_metric_formulas(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            return
        for projection in select.expressions:
            expression = projection.this if isinstance(projection, exp.Alias) else projection
            for node in _direct_expression_nodes(expression):
                for obligation in context.metric_formula_obligations:
                    if obligation.metric_index in context.metric_formula_coverage:
                        continue
                    if (
                        obligation.parse_error
                        or type(node).__name__ != obligation.signature.split(":", 1)[0]
                    ):
                        # The type check is only a cheap rejection.  The exact
                        # structural signature below remains authoritative.
                        continue
                    signature = _resolved_expression_signature(
                        node,
                        state,
                        context.allowed_columns,
                    )
                    if signature != obligation.signature:
                        continue
                    columns = _direct_columns(node)
                    resolved_origins: set[_Origin] = set()
                    unresolved = False
                    for column in columns:
                        resolved = _resolve_column(column, state, context.allowed_columns)
                        if resolved.status != "ok":
                            unresolved = True
                            break
                        resolved_origins.update(resolved.origins)
                    if unresolved:
                        continue
                    if resolved_origins and any(
                        origin.table != obligation.table
                        or origin.column not in obligation.source_columns
                        for origin in resolved_origins
                    ):
                        continue
                    if not resolved_origins and obligation.table not in state.base_tables:
                        continue
                    context.metric_formula_coverage.add(obligation.metric_index)
                    context.metric_formula_scopes.setdefault(
                        obligation.metric_index,
                        set(),
                    ).add(id(state.scope))

    def _record_entity_predicates(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            return
        roots: list[exp.Expression] = []
        for key in ("where", "having", "qualify"):
            wrapper = select.args.get(key)
            if isinstance(wrapper, exp.Expression) and isinstance(wrapper.this, exp.Expression):
                roots.append(wrapper.this)
        runtime_injected_refs = {
            item.target_field_ref
            for item in context.contract.upstream_entity_bindings
        }
        for root in roots:
            for term in _and_terms(root):
                signature = _predicate_signature(term, state, context.allowed_columns)
                if signature is None:
                    continue
                origins, operator, values = signature
                for obligation_index, obligation in enumerate(context.contract.entity_filters):
                    expected_origin = (_identifier(obligation.table), _identifier(obligation.column))
                    matching_origins = {
                        origin
                        for origin in origins
                        if (origin.table, origin.column) == expected_origin
                    }
                    if not matching_origins:
                        continue
                    if obligation.semantic_ref_id in runtime_injected_refs:
                        context.gaps.append(
                            _gap(
                                "SQL_RUNTIME_ENTITY_PREDICATE_FORBIDDEN",
                                "Core SQL may not author predicates whose values are owned by a verified entity-set artifact",
                                table=obligation.table,
                                column=obligation.column,
                                details={
                                    "semanticRefId": obligation.semantic_ref_id,
                                },
                                resolution="Remove the predicate; trusted execution injects the complete sealed IN set.",
                            )
                        )
                        continue
                    if not _operator_satisfies(operator, obligation.operator):
                        continue
                    if not _literal_values_equal(values, obligation.literal_value, obligation.operator):
                        continue
                    covered = context.entity_predicate_coverage.setdefault(obligation_index, set())
                    for origin in matching_origins:
                        covered.update(origin.scan_ids)

    def _record_time_predicates(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> None:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            return
        roots: list[exp.Expression] = []
        for key in ("where", "having", "qualify"):
            wrapper = select.args.get(key)
            if isinstance(wrapper, exp.Expression) and isinstance(wrapper.this, exp.Expression):
                roots.append(wrapper.this)
        for root in roots:
            for term in _and_terms(root):
                for origins, lower, upper in _time_predicate_signatures(
                    term,
                    state,
                    context.allowed_columns,
                ):
                    for obligation in context.time_obligations:
                        matching = {
                            origin
                            for origin in origins
                            if origin.table == obligation.table
                            and origin.column == obligation.column
                        }
                        if not matching:
                            continue
                        scan_ids = set().union(*(origin.scan_ids for origin in matching))
                        if _time_boundary_matches(lower, obligation.start_value):
                            context.time_lower_coverage.setdefault(
                                obligation.obligation_index,
                                set(),
                            ).update(scan_ids)
                        if _time_boundary_matches(upper, obligation.end_value):
                            context.time_upper_coverage.setdefault(
                                obligation.obligation_index,
                                set(),
                            ).update(scan_ids)

    def _build_scope_outputs(
        self,
        state: _ScopeState,
        context: _ValidationContext,
    ) -> dict[str, _Output]:
        select = state.scope.expression
        if not isinstance(select, exp.Select):
            # Set operations expose the output contract of their first branch.
            first_select = select.find(exp.Select) if isinstance(select, exp.Expression) else None
            if first_select is None:
                return {}
            child = next(
                (
                    item
                    for item in context.scope_states.values()
                    if item.scope.expression is first_select
                ),
                None,
            )
            return dict(child.outputs) if child else {}
        outputs: dict[str, _Output] = {}
        for index, projection in enumerate(select.expressions):
            alias = _identifier(projection.alias)
            expression = projection.this if isinstance(projection, exp.Alias) else projection
            output_name = alias
            if not output_name and isinstance(expression, exp.Column) and not isinstance(expression.this, exp.Star):
                output_name = _identifier(expression.name)
            if not output_name:
                output_name = _identifier(getattr(projection, "output_name", ""))
            if not output_name:
                output_name = "_col_%d" % index
            direct_column = expression if isinstance(expression, exp.Column) else None
            if direct_column is not None and not isinstance(direct_column.this, exp.Star):
                resolved = _resolve_column(direct_column, state, context.allowed_columns)
                if resolved.status == "ok":
                    outputs[output_name] = _Output(
                        origins=set(resolved.origins),
                        passthrough=_column_is_passthrough(direct_column, state),
                        semantic_refs=_semantic_refs_for_column(
                            direct_column,
                            state,
                            context,
                        ),
                        exact_metric_refs=_exact_metric_refs_for_column(
                            direct_column,
                            state,
                        ),
                    )
                    continue
            origins: set[_Origin] = set()
            semantic_refs: set[str] = set()
            exact_metric_refs: set[str] = set()
            for column in _direct_columns(expression):
                resolved = _resolve_column(column, state, context.allowed_columns)
                if resolved.status == "ok":
                    origins.update(resolved.origins)
                    semantic_refs.update(
                        _semantic_refs_for_column(column, state, context)
                    )
            top_signature = _resolved_expression_signature(
                expression,
                state,
                context.allowed_columns,
            )
            for node in _direct_expression_nodes(expression):
                signature = _resolved_expression_signature(
                    node,
                    state,
                    context.allowed_columns,
                )
                for obligation in context.metric_formula_obligations:
                    if not obligation.parse_error and signature == obligation.signature:
                        semantic_refs.add(obligation.semantic_ref_id)
            for obligation in context.metric_formula_obligations:
                if not obligation.parse_error and top_signature == obligation.signature:
                    exact_metric_refs.add(obligation.semantic_ref_id)
            outputs[output_name] = _Output(
                origins=origins,
                passthrough=False,
                semantic_refs=semantic_refs,
                exact_metric_refs=exact_metric_refs,
            )
        return outputs

    def _validate_entity_obligations(self, context: _ValidationContext) -> None:
        runtime_injected_refs = {
            item.target_field_ref
            for item in context.contract.upstream_entity_bindings
        }
        for index, obligation in enumerate(context.contract.entity_filters):
            if obligation.semantic_ref_id in runtime_injected_refs:
                continue
            expected_scans = context.entity_scans.get(index, set())
            covered_scans = context.entity_predicate_coverage.get(index, set())
            if expected_scans and expected_scans.issubset(covered_scans):
                continue
            details = {
                "semanticRefId": obligation.semantic_ref_id,
                "operator": obligation.operator,
                "missingScanCount": len(expected_scans - covered_scans),
            }
            context.gaps.append(
                _gap(
                    "SQL_ENTITY_PREDICATE_MISSING",
                    "Required entity literal is not a mandatory predicate on every matching table scan",
                    table=obligation.table,
                    column=obligation.column,
                    details=details,
                    resolution="Add the exact typed entity filter from GroundedQueryContract.entityFilters.",
                )
            )

    def _validate_metric_formula_obligations(self, context: _ValidationContext) -> None:
        for obligation in context.metric_formula_obligations:
            if obligation.parse_error:
                context.gaps.append(
                    _gap(
                        "SQL_GOVERNED_FORMULA_INVALID",
                        "Governed metric formula could not be parsed for AST validation",
                        table=obligation.table,
                        details={
                            "metricKey": obligation.metric_key,
                            "semanticRefId": obligation.semantic_ref_id,
                            "parserError": obligation.parse_error,
                        },
                        resolution="Repair the governed metric asset before generating SQL.",
                    )
                )
                continue
            if obligation.metric_index in context.metric_formula_coverage:
                continue
            context.gaps.append(
                _gap(
                    "SQL_METRIC_FORMULA_NOT_PRESERVED",
                    "SQL does not contain the governed metric formula as an equivalent AST subexpression",
                    table=obligation.table,
                    details={
                        "metricKey": obligation.metric_key,
                        "semanticRefId": obligation.semantic_ref_id,
                    },
                    resolution="Use the exact governed formula; CTE and window wrappers may reference its output.",
                )
            )

    def _validate_metric_fanout(self, context: _ValidationContext) -> None:
        if not context.metric_formula_obligations:
            return
        states_by_scope = {
            id(state.scope): state
            for state in context.scope_states.values()
        }
        for proof, join_state in context.join_proofs:
            policy = re.sub(
                r"[^A-Z0-9]+",
                "_",
                str(proof.relationship.fanout_policy or "").upper(),
            ).strip("_")
            cardinality = _normalize_cardinality(proof.relationship.cardinality)
            for obligation in context.metric_formula_obligations:
                formula_states = [
                    states_by_scope[scope_id]
                    for scope_id in context.metric_formula_scopes.get(
                        obligation.metric_index,
                        set(),
                    )
                    if scope_id in states_by_scope
                ]
                if not any(
                    _scope_feeds(join_state.scope, formula_state.scope)
                    for formula_state in formula_states
                ):
                    continue
                all_join_tables = set(proof.left_tables) | set(proof.right_tables)
                policy_blocks = (
                    obligation.table in all_join_tables
                    and any(token in policy for token in ("FORBID", "BLOCK", "UNSAFE"))
                )
                duplicated_tables = _fanout_duplicated_tables(proof, cardinality)
                cardinality_blocks = obligation.table in duplicated_tables
                if not policy_blocks and not cardinality_blocks:
                    continue
                context.gaps.append(
                    _gap(
                        "SQL_METRIC_FANOUT_UNSAFE",
                        "A pre-aggregation join can multiply rows used by a governed metric",
                        table=obligation.table,
                        relationship_ref_id=proof.relationship_ref_id,
                        details={
                            "metricKey": obligation.metric_key,
                            "cardinality": proof.relationship.cardinality,
                            "fanoutPolicy": proof.relationship.fanout_policy,
                            "relationshipName": proof.relationship_name,
                        },
                        resolution="Aggregate before the join or select a relationship direction/policy that preserves the metric grain.",
                    )
                )

    def _validate_time_obligations(self, context: _ValidationContext) -> None:
        for obligation in context.time_obligations:
            expected = context.time_scans.get(obligation.obligation_index, set())
            lower = context.time_lower_coverage.get(obligation.obligation_index, set())
            upper = context.time_upper_coverage.get(obligation.obligation_index, set())
            if expected and expected.issubset(lower) and expected.issubset(upper):
                continue
            context.gaps.append(
                _gap(
                    "SQL_TIME_PREDICATE_MISSING",
                    "Explicit contract time range is not a mandatory bounded predicate on every relevant table scan",
                    table=obligation.table,
                    column=obligation.column,
                    details={
                        "startValue": obligation.start_value,
                        "endValue": obligation.end_value,
                        "missingLowerScanCount": len(expected - lower),
                        "missingUpperScanCount": len(expected - upper),
                    },
                    resolution="Apply both grounded time boundaries on the governed time column outside OR/NOT branches.",
                )
            )

    def _validate_final_outputs(self, context: _ValidationContext) -> None:
        required: list[tuple[str, str, str, str]] = []
        required.extend(
            (metric.metric_key, metric.semantic_ref_id, metric.table, "METRIC")
            for metric in context.contract.metrics
        )
        required.extend(
            (
                field.output_alias or field.column,
                field.semantic_ref_id,
                field.table,
                "SELECTED_FIELD",
            )
            for field in context.contract.selected_fields
        )
        required.extend(
            (dimension.column, dimension.semantic_ref_id, dimension.table, "DIMENSION")
            for dimension in context.contract.dimensions
        )
        normalized_required = [_identifier(item[0]) for item in required]
        duplicate_contract_aliases = sorted(
            {
                alias
                for alias in normalized_required
                if normalized_required.count(alias) > 1
            }
        )
        if duplicate_contract_aliases:
            context.gaps.append(
                _gap(
                    "SQL_CONTRACT_OUTPUT_ALIAS_CONFLICT",
                    "Grounded contract contains duplicate final output aliases",
                    details={"aliases": duplicate_contract_aliases},
                    resolution="Assign a unique governed output alias to every output binding.",
                )
            )
        duplicate_sql_aliases = sorted(
            {
                alias
                for alias in context.final_output_columns
                if context.final_output_columns.count(alias) > 1
            }
        )
        if duplicate_sql_aliases:
            context.gaps.append(
                _gap(
                    "SQL_OUTPUT_ALIAS_DUPLICATE",
                    "Final SELECT output aliases must be unique",
                    details={"aliases": duplicate_sql_aliases},
                    resolution="Give every final projection one unique contract-defined alias.",
                )
            )
        expected_set = set(normalized_required)
        actual_set = set(context.final_output_columns)
        if expected_set != actual_set or len(context.final_output_columns) != len(expected_set):
            context.gaps.append(
                _gap(
                    "SQL_OUTPUT_SET_MISMATCH",
                    "Final SELECT output set must exactly match the grounded contract",
                    details={
                        "missingAliases": sorted(expected_set - actual_set),
                        "extraAliases": sorted(actual_set - expected_set),
                    },
                    resolution="Return every and only the contract-bound output aliases.",
                )
            )
        for raw_alias, semantic_ref, table, binding_kind in required:
            alias = _identifier(raw_alias)
            output = context.final_outputs.get(alias)
            if output is not None and semantic_ref in output.semantic_refs:
                if binding_kind == "METRIC":
                    if semantic_ref in output.exact_metric_refs:
                        continue
                    context.gaps.append(
                        _gap(
                            "SQL_METRIC_OUTPUT_EXPRESSION_MISMATCH",
                            "Metric final alias must be the governed formula itself or a pure CTE passthrough",
                            table=table,
                            column=raw_alias,
                            details={"semanticRefId": semantic_ref},
                            resolution="Remove CASE/arithmetic/cast wrappers from the metric's final expression.",
                        )
                    )
                    continue
                expected_origin = (_identifier(table), _identifier(raw_alias))
                binding = next(
                    (
                        item
                        for item in [
                            *context.contract.selected_fields,
                            *context.contract.dimensions,
                        ]
                        if item.semantic_ref_id == semantic_ref
                    ),
                    None,
                )
                if binding is not None:
                    expected_origin = (
                        _identifier(binding.table),
                        _identifier(binding.column),
                    )
                if output.passthrough and any(
                    (origin.table, origin.column) == expected_origin
                    for origin in output.origins
                ):
                    continue
            context.gaps.append(
                _gap(
                    "SQL_OUTPUT_BINDING_MISSING",
                    "Final SQL output does not preserve the required alias-to-semantic-binding identity",
                    table=table,
                    column=raw_alias,
                    details={
                        "outputAlias": raw_alias,
                        "semanticRefId": semantic_ref,
                        "bindingKind": binding_kind,
                    },
                    resolution="Project the bound expression under its contract-defined final output alias.",
                )
            )


def grounded_query_contract_fingerprint(contract: GroundedQueryContract) -> str:
    """Fingerprint only the semantic authority relevant to SQL generation."""

    payload = {
        "version": contract.contract_version,
        "status": contract.status,
        "queryShape": contract.query_shape,
        "primaryTable": contract.primary_table,
        "tables": [
            {
                "topic": item.topic,
                "table": item.table,
                "timeColumn": item.time_column,
                "merchantFilterColumn": item.merchant_filter_column,
                "detailRefId": item.detail_ref_id,
            }
            for item in contract.tables
        ],
        "metrics": [item.model_dump(by_alias=True, mode="json") for item in contract.metrics],
        "dimensions": [item.model_dump(by_alias=True, mode="json") for item in contract.dimensions],
        "selectedFields": [item.model_dump(by_alias=True, mode="json") for item in contract.selected_fields],
        "entityFilters": [item.model_dump(by_alias=True, mode="json") for item in contract.entity_filters],
        "upstreamEntityBindings": [
            item.model_dump(by_alias=True, mode="json")
            for item in contract.upstream_entity_bindings
        ],
        "relationships": [item.model_dump(by_alias=True, mode="json") for item in contract.relationships],
        "timeRange": contract.time_range.model_dump(by_alias=True, mode="json"),
        "evidenceRefs": sorted(contract.evidence_refs),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_context(contract: GroundedQueryContract) -> _ValidationContext:
    trusted_refs = set(contract.evidence_refs)
    trusted_refs.update(str(item.ref_id) for item in contract.evidence)
    allowed_tables: dict[str, str] = {}
    allowed_columns: dict[str, set[str]] = {}
    merchant_columns: dict[str, str] = {}
    column_binding_refs: dict[tuple[str, str], set[str]] = {}
    for binding in contract.tables:
        table = _identifier(binding.table)
        if not table or not binding.detail_ref_id or binding.detail_ref_id not in trusted_refs:
            continue
        allowed_tables[table] = binding.table
        allowed_columns.setdefault(table, set())
        if binding.time_column:
            allowed_columns[table].add(_identifier(binding.time_column))
        if binding.merchant_filter_column:
            merchant = _identifier(binding.merchant_filter_column)
            allowed_columns[table].add(merchant)
            merchant_columns[table] = merchant

    metric_formula_obligations: list[_MetricFormulaObligation] = []
    for metric_index, metric in enumerate(contract.metrics):
        if metric.semantic_ref_id not in trusted_refs:
            continue
        table = _identifier(metric.table)
        allowed_columns.setdefault(table, set()).update(
            _identifier(item) for item in metric.source_columns if _identifier(item)
        )
        if metric.time_column:
            allowed_columns.setdefault(table, set()).add(_identifier(metric.time_column))
        signature, parse_error = _governed_formula_signature(metric.formula)
        metric_formula_obligations.append(
            _MetricFormulaObligation(
                metric_index=metric_index,
                semantic_ref_id=metric.semantic_ref_id,
                metric_key=metric.metric_key,
                table=table,
                source_columns=frozenset(
                    _identifier(item)
                    for item in metric.source_columns
                    if _identifier(item)
                ),
                signature=signature,
                parse_error=parse_error,
            )
        )
    for binding in [*contract.dimensions, *contract.selected_fields, *contract.entity_filters]:
        if binding.semantic_ref_id not in trusted_refs:
            continue
        table = _identifier(binding.table)
        column = _identifier(binding.column)
        if table and column:
            allowed_columns.setdefault(table, set()).add(column)
            column_binding_refs.setdefault((table, column), set()).add(
                binding.semantic_ref_id
            )

    relationships: list[GroundedRelationshipBinding] = []
    for relationship in contract.relationships:
        if relationship.semantic_ref_id not in trusted_refs:
            continue
        relationships.append(relationship)
        left_table = _identifier(relationship.left_table)
        right_table = _identifier(relationship.right_table)
        for pair in relationship.keys:
            if len(pair) != 2:
                continue
            allowed_columns.setdefault(left_table, set()).add(_identifier(pair[0]))
            allowed_columns.setdefault(right_table, set()).add(_identifier(pair[1]))
    time_obligations: list[_TimeObligation] = []
    if bool(contract.time_range.explicit):
        table_bindings = {
            _identifier(item.table): item
            for item in contract.tables
            if _identifier(item.table)
        }
        relevant: list[tuple[str, str]] = []
        for metric in contract.metrics:
            table = _identifier(metric.table)
            table_binding = table_bindings.get(table)
            column = _identifier(
                metric.time_column
                or (table_binding.time_column if table_binding else "")
            )
            if table and column:
                relevant.append((table, column))
        primary_table = _identifier(contract.primary_table)
        primary_binding = table_bindings.get(primary_table)
        if primary_binding and primary_binding.time_column:
            relevant.append((primary_table, _identifier(primary_binding.time_column)))
        for entity_filter in contract.entity_filters:
            table = _identifier(entity_filter.table)
            binding = table_bindings.get(table)
            if binding and binding.time_column:
                relevant.append((table, _identifier(binding.time_column)))
        start_value = str(
            contract.time_range.execution_start_value
            or contract.time_range.execution_start_date
            or contract.time_range.start_date
            or ""
        ).strip()
        end_value = str(
            contract.time_range.execution_end_value
            or contract.time_range.execution_end_date
            or contract.time_range.end_date
            or ""
        ).strip()
        for table, column in _dedupe_pairs(relevant):
            time_obligations.append(
                _TimeObligation(
                    obligation_index=len(time_obligations),
                    table=table,
                    column=column,
                    start_value=start_value,
                    end_value=end_value,
                )
            )

    return _ValidationContext(
        contract=contract,
        trusted_refs=trusted_refs,
        allowed_tables=allowed_tables,
        allowed_columns=allowed_columns,
        merchant_columns=merchant_columns,
        relationships=relationships,
        metric_formula_obligations=metric_formula_obligations,
        time_obligations=time_obligations,
        column_binding_refs=column_binding_refs,
    )


def _resolve_column(
    column: exp.Column,
    state: _ScopeState,
    allowed_columns: Mapping[str, set[str]],
) -> _ResolvedColumn:
    name = _identifier(column.name)
    qualifier = _identifier(column.table)
    if not name:
        return _ResolvedColumn(status="unknown")
    if qualifier:
        source = state.sources.get(qualifier)
        if source is None:
            return _ResolvedColumn(status="unknown")
        return _resolve_from_source(source, name, allowed_columns)
    matches: list[_ResolvedColumn] = []
    for source in state.sources.values():
        resolved = _resolve_from_source(source, name, allowed_columns)
        if resolved.status == "ok":
            matches.append(resolved)
    if not matches:
        return _ResolvedColumn(status="unknown")
    if len(matches) > 1:
        return _ResolvedColumn(status="ambiguous")
    return matches[0]


def _resolve_from_source(
    source: _Source,
    name: str,
    allowed_columns: Mapping[str, set[str]],
) -> _ResolvedColumn:
    if source.base_table:
        if name not in allowed_columns.get(source.base_table, set()):
            return _ResolvedColumn(status="unknown", source_alias=source.alias)
        return _ResolvedColumn(
            status="ok",
            origins={
                _Origin(
                    table=source.base_table,
                    column=name,
                    scan_ids=frozenset({source.scan_id}),
                )
            },
            source_alias=source.alias,
        )
    output = source.outputs.get(name)
    if output is None:
        return _ResolvedColumn(status="unknown", source_alias=source.alias)
    return _ResolvedColumn(
        status="ok",
        origins=set(output.origins),
        source_alias=source.alias,
        derived_output=True,
    )


def _semantic_refs_for_column(
    column: exp.Column,
    state: _ScopeState,
    context: _ValidationContext,
) -> set[str]:
    qualifier = _identifier(column.table)
    name = _identifier(column.name)
    candidates: list[_Source] = []
    if qualifier:
        source = state.sources.get(qualifier)
        if source is not None:
            candidates = [source]
    else:
        for source in state.sources.values():
            if source.base_table and name in context.allowed_columns.get(source.base_table, set()):
                candidates.append(source)
            elif source.derived and name in source.outputs:
                candidates.append(source)
    if len(candidates) != 1:
        return set()
    source = candidates[0]
    if source.base_table:
        return set(context.column_binding_refs.get((source.base_table, name), set()))
    output = source.outputs.get(name)
    return set(output.semantic_refs) if output else set()


def _column_is_passthrough(column: exp.Column, state: _ScopeState) -> bool:
    qualifier = _identifier(column.table)
    name = _identifier(column.name)
    candidates: list[_Source] = []
    if qualifier:
        source = state.sources.get(qualifier)
        if source is not None:
            candidates = [source]
    else:
        candidates = [
            source
            for source in state.sources.values()
            if source.base_table or name in source.outputs
        ]
    if len(candidates) != 1:
        return False
    source = candidates[0]
    if source.base_table:
        return True
    output = source.outputs.get(name)
    return bool(output and output.passthrough)


def _exact_metric_refs_for_column(
    column: exp.Column,
    state: _ScopeState,
) -> set[str]:
    qualifier = _identifier(column.table)
    name = _identifier(column.name)
    candidates: list[_Source] = []
    if qualifier:
        source = state.sources.get(qualifier)
        if source is not None:
            candidates = [source]
    else:
        candidates = [
            source
            for source in state.sources.values()
            if source.derived and name in source.outputs
        ]
    if len(candidates) != 1 or candidates[0].base_table:
        return set()
    output = candidates[0].outputs.get(name)
    return set(output.exact_metric_refs) if output else set()


def _source_column_origins(
    source: _Source,
    column: str,
    allowed_columns: Mapping[str, set[str]],
) -> set[_Origin]:
    if source.base_table:
        if column not in allowed_columns.get(source.base_table, set()):
            return set()
        return {
            _Origin(
                table=source.base_table,
                column=column,
                scan_ids=frozenset({source.scan_id}),
            )
        }
    output = source.outputs.get(column)
    if output is None or not output.passthrough:
        return set()
    return set(output.origins)


def _scan_table(scan_id: str) -> str:
    return str(scan_id or "").rsplit(":", 1)[-1]


def _prove_join(
    state: _ScopeState,
    left_aliases: Sequence[str],
    right_alias: str,
    join: exp.Join,
    context: _ValidationContext,
) -> _JoinProof | None:
    left_sources = [state.sources[item] for item in left_aliases if item in state.sources]
    right_source = state.sources[right_alias]
    left_tables = set().union(*(item.base_tables for item in left_sources)) if left_sources else set()
    right_tables = set(right_source.base_tables)
    left_scan_ids = set().union(*(item.scan_ids for item in left_sources)) if left_sources else set()
    right_scan_ids = set(right_source.scan_ids)
    if not left_tables or not right_tables or not left_scan_ids or not right_scan_ids:
        return None
    # Each edge retains the concrete physical scan on both sides.  Table-only
    # pairs are insufficient when one table is joined more than once: keys
    # from an already-connected alias must not satisfy the current right scan.
    actual_edges: set[tuple[str, str, str, str, str, str]] = set()
    on = join.args.get("on")
    if on is not None:
        for term in _and_terms(on):
            predicate = _unwrap(term)
            if not isinstance(predicate, exp.EQ):
                continue
            if not isinstance(predicate.this, exp.Column) or not isinstance(predicate.expression, exp.Column):
                continue
            if not _column_is_passthrough(predicate.this, state) or not _column_is_passthrough(
                predicate.expression,
                state,
            ):
                continue
            left = _resolve_column(predicate.this, state, context.allowed_columns)
            right = _resolve_column(predicate.expression, state, context.allowed_columns)
            if left.status != "ok" or right.status != "ok":
                continue
            for first in left.origins:
                for second in right.origins:
                    for first_scan in first.scan_ids:
                        for second_scan in second.scan_ids:
                            if first_scan in left_scan_ids and second_scan in right_scan_ids:
                                actual_edges.add(
                                    (
                                        first_scan,
                                        first.table,
                                        first.column,
                                        second_scan,
                                        second.table,
                                        second.column,
                                    )
                                )
                            elif second_scan in left_scan_ids and first_scan in right_scan_ids:
                                actual_edges.add(
                                    (
                                        second_scan,
                                        second.table,
                                        second.column,
                                        first_scan,
                                        first.table,
                                        first.column,
                                    )
                                )
    for identifier in join.args.get("using") or []:
        column = _identifier(getattr(identifier, "name", "") or getattr(identifier, "this", ""))
        for left_source in left_sources:
            for first in _source_column_origins(
                left_source,
                column,
                context.allowed_columns,
            ):
                for second in _source_column_origins(
                    right_source,
                    column,
                    context.allowed_columns,
                ):
                    for first_scan in first.scan_ids:
                        for second_scan in second.scan_ids:
                            actual_edges.add(
                                (
                                    first_scan,
                                    first.table,
                                    first.column,
                                    second_scan,
                                    second.table,
                                    second.column,
                                )
                            )

    candidates: list[
        tuple[
            GroundedRelationshipBinding,
            bool,
            str,
            str,
            set[tuple[str, str, str, str, str, str]],
        ]
    ] = []
    for relationship in context.relationships:
        rel_left = _identifier(relationship.left_table)
        rel_right = _identifier(relationship.right_table)
        for left_scan in left_scan_ids:
            left_scan_table = _scan_table(left_scan)
            for right_scan in right_scan_ids:
                right_scan_table = _scan_table(right_scan)
                forward = left_scan_table == rel_left and right_scan_table == rel_right
                reverse = left_scan_table == rel_right and right_scan_table == rel_left
                if not forward and not reverse:
                    continue
                expected: set[tuple[str, str, str, str, str, str]] = set()
                for pair in relationship.keys:
                    if len(pair) != 2:
                        expected.clear()
                        break
                    if forward:
                        expected.add(
                            (
                                left_scan,
                                rel_left,
                                _identifier(pair[0]),
                                right_scan,
                                rel_right,
                                _identifier(pair[1]),
                            )
                        )
                    else:
                        expected.add(
                            (
                                left_scan,
                                rel_right,
                                _identifier(pair[1]),
                                right_scan,
                                rel_left,
                                _identifier(pair[0]),
                            )
                        )
                if not expected or not expected.issubset(actual_edges):
                    continue
                expected_type = _normalize_join_type(relationship.join_type)
                actual_type = _join_type(join)
                if expected_type:
                    if reverse:
                        expected_type = _reverse_join_type(expected_type)
                    if actual_type != expected_type:
                        continue
                candidates.append(
                    (relationship, bool(forward), left_scan, right_scan, expected)
                )
    if len(candidates) != 1:
        return None
    relationship, forward, _left_scan, _right_scan, allowed_edges = candidates[0]
    # Every cross-source equality in this JOIN must belong to the one proven
    # relationship for the current right scan.  This prevents mixing keys from
    # different aliases or stitching together two partial relationship proofs.
    if not actual_edges.issubset(allowed_edges):
        return None
    return _JoinProof(
        relationship_ref_id=relationship.semantic_ref_id,
        relationship_name=relationship.name,
        relationship=relationship,
        forward=forward,
        left_scan_ids=frozenset(left_scan_ids),
        right_scan_ids=frozenset(right_scan_ids),
        left_tables=frozenset(left_tables),
        right_tables=frozenset(right_tables),
    )


def _predicate_signature(
    expression: exp.Expression,
    state: _ScopeState,
    allowed_columns: Mapping[str, set[str]],
) -> tuple[set[_Origin], str, list[tuple[str, str]]] | None:
    predicate = _unwrap(expression)
    operator = ""
    column: exp.Column | None = None
    value_nodes: list[exp.Expression] = []
    reverse = False
    if isinstance(predicate, exp.In) and isinstance(predicate.this, exp.Column):
        if predicate.args.get("query") is not None:
            return None
        operator = "IN"
        column = predicate.this
        value_nodes = list(predicate.expressions or [])
    else:
        operators: list[tuple[type[exp.Expression], str]] = [
            (exp.EQ, "EQ"),
            (exp.NEQ, "NE"),
            (exp.GT, "GT"),
            (exp.GTE, "GTE"),
            (exp.LT, "LT"),
            (exp.LTE, "LTE"),
            (exp.Like, "LIKE"),
        ]
        operator = next((name for cls, name in operators if isinstance(predicate, cls)), "")
        if not operator:
            return None
        if isinstance(predicate.this, exp.Column):
            column = predicate.this
            value_nodes = [predicate.expression]
        elif isinstance(predicate.expression, exp.Column):
            column = predicate.expression
            value_nodes = [predicate.this]
            reverse = True
        else:
            return None
    values = [_sql_literal_token(item) for item in value_nodes]
    if not values or any(item is None for item in values):
        return None
    resolved = _resolve_column(column, state, allowed_columns)
    if resolved.status != "ok":
        return None
    if reverse:
        operator = {"GT": "LT", "GTE": "LTE", "LT": "GT", "LTE": "GTE"}.get(operator, operator)
    return resolved.origins, operator, [item for item in values if item is not None]


def _governed_formula_signature(formula: str) -> tuple[str, str]:
    text = str(formula or "").strip()
    if not text:
        return "", "formula is empty"
    try:
        parsed = sqlglot.parse_one("SELECT %s AS __metric_value" % text, read="doris")
    except Exception as exc:
        return "", str(exc)[:500]
    if not isinstance(parsed, exp.Select) or len(parsed.expressions) != 1:
        return "", "formula did not parse as one SQL expression"
    projection = parsed.expressions[0]
    expression = projection.this if isinstance(projection, exp.Alias) else projection
    return _unqualified_expression_signature(expression), ""


def _resolved_expression_signature(
    expression: exp.Expression,
    state: _ScopeState,
    allowed_columns: Mapping[str, set[str]],
) -> str:
    def normalize(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
            resolved = _resolve_column(node, state, allowed_columns)
            source_names = {origin.column for origin in resolved.origins}
            name = next(iter(source_names)) if len(source_names) == 1 else _identifier(node.name)
            return exp.Column(this=exp.Identifier(this=name, quoted=False))
        if isinstance(node, exp.Identifier):
            return exp.Identifier(this=_identifier(node.this), quoted=False)
        return node

    normalized = _unwrap(expression).copy().transform(normalize)
    return "%s:%s" % (
        type(normalized).__name__,
        normalized.sql(dialect="doris", pretty=False, normalize=True, comments=False),
    )


def _unqualified_expression_signature(expression: exp.Expression) -> str:
    def normalize(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Column) and not isinstance(node.this, exp.Star):
            return exp.Column(
                this=exp.Identifier(this=_identifier(node.name), quoted=False)
            )
        if isinstance(node, exp.Identifier):
            return exp.Identifier(this=_identifier(node.this), quoted=False)
        return node

    normalized = _unwrap(expression).copy().transform(normalize)
    return "%s:%s" % (
        type(normalized).__name__,
        normalized.sql(dialect="doris", pretty=False, normalize=True, comments=False),
    )


def _time_predicate_signatures(
    expression: exp.Expression,
    state: _ScopeState,
    allowed_columns: Mapping[str, set[str]],
) -> list[tuple[set[_Origin], tuple[str, str] | None, tuple[str, str] | None]]:
    predicate = _unwrap(expression)
    if isinstance(predicate, exp.Between) and isinstance(predicate.this, exp.Column):
        resolved = _resolve_column(predicate.this, state, allowed_columns)
        if resolved.status != "ok":
            return []
        return [
            (
                resolved.origins,
                _sql_literal_token(predicate.args.get("low")),
                _sql_literal_token(predicate.args.get("high")),
            )
        ]
    operators: list[tuple[type[exp.Expression], str]] = [
        (exp.EQ, "EQ"),
        (exp.GT, "GT"),
        (exp.GTE, "GTE"),
        (exp.LT, "LT"),
        (exp.LTE, "LTE"),
    ]
    operator = next((name for cls, name in operators if isinstance(predicate, cls)), "")
    if not operator:
        return []
    column: exp.Column | None = None
    literal: exp.Expression | None = None
    reverse = False
    if isinstance(predicate.this, exp.Column):
        column = predicate.this
        literal = predicate.expression
    elif isinstance(predicate.expression, exp.Column):
        column = predicate.expression
        literal = predicate.this
        reverse = True
    if column is None or literal is None:
        return []
    token = _sql_literal_token(literal)
    if token is None:
        return []
    resolved = _resolve_column(column, state, allowed_columns)
    if resolved.status != "ok":
        return []
    if reverse:
        operator = {"GT": "LT", "GTE": "LTE", "LT": "GT", "LTE": "GTE"}.get(
            operator,
            operator,
        )
    if operator == "EQ":
        return [(resolved.origins, token, token)]
    if operator in {"GT", "GTE"}:
        return [(resolved.origins, token, None)]
    return [(resolved.origins, None, token)]


def _time_boundary_matches(
    actual: tuple[str, str] | None,
    expected: str,
) -> bool:
    if actual is None:
        return False
    if not expected:
        return True
    return actual[1] == str(expected)


def _operator_satisfies(actual: str, expected: str) -> bool:
    return _normalize_operator(actual) == _normalize_operator(expected)


def _literal_values_equal(
    actual: Sequence[tuple[str, str]],
    expected: Any,
    operator: str,
) -> bool:
    if _normalize_operator(operator) == "IN":
        expected_values = expected if isinstance(expected, (list, tuple, set)) else [expected]
    else:
        expected_values = [expected]
    expected_tokens = [_python_literal_token(item) for item in expected_values]
    return sorted(actual) == sorted(expected_tokens)


def _sql_literal_token(expression: exp.Expression) -> tuple[str, str] | None:
    current = _unwrap(expression)
    if isinstance(current, exp.Cast):
        current = _unwrap(current.this)
    if isinstance(current, exp.Literal):
        if current.is_string:
            return "string", str(current.this)
        return "number", _canonical_number(current.this)
    if isinstance(current, exp.Boolean):
        return "boolean", "true" if bool(current.this) else "false"
    if isinstance(current, exp.Null):
        return "null", "null"
    return None


def _python_literal_token(value: Any) -> tuple[str, str]:
    if value is None:
        return "null", "null"
    if isinstance(value, bool):
        return "boolean", "true" if value else "false"
    if isinstance(value, (int, float, Decimal)):
        return "number", _canonical_number(value)
    return "string", str(value)


def _canonical_number(value: Any) -> str:
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return str(value)
    if decimal == decimal.to_integral():
        return str(decimal.quantize(Decimal(1)))
    return format(decimal.normalize(), "f")


def _normalize_candidate(
    candidate: GroundedSqlCandidate | Mapping[str, Any] | str,
) -> GroundedSqlCandidate:
    if isinstance(candidate, GroundedSqlCandidate):
        return candidate
    if isinstance(candidate, str):
        return GroundedSqlCandidate(sql=candidate)
    return GroundedSqlCandidate.model_validate(candidate)


def _canonical_sql(parsed: exp.Expression, dialect: str) -> str:
    def normalize_identifier(node: exp.Expression) -> exp.Expression:
        if not isinstance(node, exp.Identifier):
            return node
        return exp.Identifier(this=_identifier(node.this), quoted=False)

    normalized = parsed.copy().transform(normalize_identifier)
    return normalized.sql(
        dialect=dialect,
        pretty=False,
        normalize=True,
        comments=False,
    ).rstrip(";")


def _ast_fingerprint(canonical_sql: str, dialect: str) -> str:
    parsed = sqlglot.parse_one(canonical_sql, read=dialect)
    dump = parsed.dump()
    structural = [
        {key: value for key, value in item.items() if key != "m"}
        for item in dump
    ]
    encoded = json.dumps(structural, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _direct_columns(expression: exp.Expression) -> list[exp.Column]:
    columns: list[exp.Column] = []

    def visit(node: exp.Expression, root: bool = False) -> None:
        if not root and isinstance(node, (exp.Select, exp.Subquery, exp.CTE)):
            return
        if isinstance(node, exp.Column):
            if not isinstance(node.this, exp.Star):
                columns.append(node)
            return
        for child in node.iter_expressions():
            visit(child)

    visit(expression, root=True)
    return columns


def _direct_expression_nodes(expression: exp.Expression) -> list[exp.Expression]:
    nodes: list[exp.Expression] = []

    def visit(node: exp.Expression, root: bool = False) -> None:
        if not root and isinstance(node, (exp.Select, exp.Subquery, exp.CTE)):
            return
        nodes.append(node)
        for child in node.iter_expressions():
            visit(child)

    visit(expression, root=True)
    return nodes


def _select_output_names(expression: exp.Expression) -> list[str]:
    if not isinstance(expression, exp.Select):
        return []
    output: list[str] = []
    for index, projection in enumerate(expression.expressions):
        alias = _identifier(projection.alias)
        target = projection.this if isinstance(projection, exp.Alias) else projection
        name = alias
        if not name and isinstance(target, exp.Column) and not isinstance(target.this, exp.Star):
            name = _identifier(target.name)
        if not name:
            name = _identifier(getattr(projection, "output_name", ""))
        output.append(name or "_col_%d" % index)
    return output


def _and_terms(expression: exp.Expression) -> list[exp.Expression]:
    current = _unwrap(expression)
    if isinstance(current, exp.And):
        return _and_terms(current.this) + _and_terms(current.expression)
    return [current]


def _unwrap(expression: exp.Expression) -> exp.Expression:
    current = expression
    while isinstance(current, exp.Paren):
        current = current.this
    return current


def _relation_alias(expression: exp.Expression) -> str:
    if isinstance(expression, exp.Table):
        return _identifier(expression.alias_or_name)
    return _identifier(getattr(expression, "alias_or_name", ""))


def _join_type(join: exp.Join) -> str:
    kind = _identifier(join.args.get("kind")).upper()
    side = _identifier(join.args.get("side")).upper()
    method = _identifier(join.args.get("method")).upper()
    if kind == "CROSS" or method == "CROSS":
        return "CROSS"
    if kind in {"SEMI", "ANTI"}:
        return "%s_%s" % (side or "LEFT", kind)
    if side:
        return side
    if kind and kind != "JOIN":
        return kind
    return "INNER"


def _normalize_join_type(value: Any) -> str:
    text = re.sub(r"\s+", "_", str(value or "").strip().upper())
    text = text.removesuffix("_JOIN").removesuffix("JOIN").strip("_")
    if text in {"", "DEFAULT"}:
        return ""
    if text == "OUTER":
        return "FULL"
    if text in {"LEFT_OUTER", "RIGHT_OUTER", "FULL_OUTER"}:
        return text.split("_", 1)[0]
    return text


def _reverse_join_type(value: str) -> str:
    return {"LEFT": "RIGHT", "RIGHT": "LEFT"}.get(value, value)


def _normalize_cardinality(value: Any) -> str:
    text = re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")
    return {
        "1_N": "ONE_TO_MANY",
        "1_M": "ONE_TO_MANY",
        "ONE_MANY": "ONE_TO_MANY",
        "N_1": "MANY_TO_ONE",
        "M_1": "MANY_TO_ONE",
        "MANY_ONE": "MANY_TO_ONE",
        "1_1": "ONE_TO_ONE",
        "ONE_ONE": "ONE_TO_ONE",
        "N_N": "MANY_TO_MANY",
        "M_N": "MANY_TO_MANY",
        "N_M": "MANY_TO_MANY",
        "M_M": "MANY_TO_MANY",
        "MANY_MANY": "MANY_TO_MANY",
    }.get(text, text)


def _fanout_duplicated_tables(
    proof: _JoinProof,
    cardinality: str,
) -> set[str]:
    if cardinality == "MANY_TO_MANY":
        return set(proof.left_tables) | set(proof.right_tables)
    if cardinality == "ONE_TO_MANY":
        return set(proof.left_tables if proof.forward else proof.right_tables)
    if cardinality == "MANY_TO_ONE":
        return set(proof.right_tables if proof.forward else proof.left_tables)
    return set()


def _scope_feeds(source_scope: Any, target_scope: Any) -> bool:
    current = source_scope
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        if current is target_scope:
            return True
        seen.add(id(current))
        current = getattr(current, "parent", None)
    return False


def _normalize_operator(value: Any) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    return {
        "=": "EQ",
        "==": "EQ",
        "EQUAL": "EQ",
        "EQUALS": "EQ",
        "!=": "NE",
        "<>": "NE",
        "NEQ": "NE",
        ">": "GT",
        ">=": "GTE",
        "<": "LT",
        "<=": "LTE",
    }.get(text, text)


def _identifier(value: Any) -> str:
    text = str(value or "").strip().strip("`").strip('"')
    return text.casefold()


def _table_qualifier(table: exp.Table) -> str:
    parts = [str(table.catalog or "").strip(), str(table.db or "").strip()]
    return ".".join(item for item in parts if item)


def _dedupe_pairs(items: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    output: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        output.append(item)
    return output


def _gap(
    code: str,
    message: str,
    *,
    table: str = "",
    column: str = "",
    relationship_ref_id: str = "",
    resolution: str = "",
    details: Mapping[str, Any] | None = None,
) -> GroundedSqlValidationGap:
    return GroundedSqlValidationGap(
        code=code,
        message=message,
        table=table,
        column=column,
        relationship_ref_id=relationship_ref_id,
        resolution=resolution,
        details=dict(details or {}),
    )


def _dedupe_gaps(
    gaps: Iterable[GroundedSqlValidationGap],
) -> list[GroundedSqlValidationGap]:
    output: list[GroundedSqlValidationGap] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for gap in gaps:
        identity = (
            gap.code,
            gap.message,
            gap.table,
            gap.column,
            json.dumps(gap.details, ensure_ascii=False, sort_keys=True, default=str),
        )
        if identity in seen:
            continue
        seen.add(identity)
        output.append(gap)
    return output


__all__ = [
    "GroundedSqlCandidate",
    "GroundedSqlCandidateValidator",
    "GroundedSqlValidationGap",
    "GroundedSqlValidationResult",
    "grounded_query_contract_fingerprint",
]
