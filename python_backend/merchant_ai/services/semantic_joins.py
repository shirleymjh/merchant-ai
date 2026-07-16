from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple

from merchant_ai.models import PlanningAssetPack, RelationshipEntry


JoinUsage = Literal["detail", "filter_scope", "aggregate"]


@dataclass(frozen=True)
class JoinPlanningGap:
    """A fail-closed reason that prevents a governed relationship plan."""

    code: str
    reason: str
    relationship_ref_id: str = ""
    tables: Tuple[str, ...] = ()
    evidence: str = ""


@dataclass(frozen=True)
class JoinKeyBinding:
    from_column: str
    to_column: str


@dataclass(frozen=True)
class JoinStep:
    relationship_ref_id: str
    relationship_id: str
    from_table: str
    to_table: str
    keys: Tuple[JoinKeyBinding, ...]
    strategy: str
    cardinality: str = ""
    fanout_safe: Optional[bool] = None
    join_type: str = ""
    base_rows_preserved: Optional[bool] = None
    safety_evidence: str = ""


@dataclass(frozen=True)
class JoinPlan:
    base_table: str
    required_tables: Tuple[str, ...]
    usage: JoinUsage
    execution_strategy: str
    steps: Tuple[JoinStep, ...]

    @property
    def relationship_ref_ids(self) -> Tuple[str, ...]:
        return tuple(step.relationship_ref_id for step in self.steps)


@dataclass(frozen=True)
class JoinPlanningResult:
    plan: Optional[JoinPlan] = None
    gaps: Tuple[JoinPlanningGap, ...] = ()

    @property
    def valid(self) -> bool:
        return self.plan is not None and not self.gaps


@dataclass(frozen=True)
class _GovernedEdge:
    ref_id: str
    relationship: RelationshipEntry
    left_columns: Tuple[str, ...]
    right_columns: Tuple[str, ...]

    @property
    def left_table(self) -> str:
        return self.relationship.left_table

    @property
    def right_table(self) -> str:
        return self.relationship.right_table


class GovernedJoinPlanner:
    """Plans relationship paths using only recalled, governed relationship assets.

    ``relationship_ref_ids=None`` lets the planner find a unique minimum path from
    the recalled relationship graph. Supplying refs makes them an exact allow-list;
    every supplied edge must be necessary for the selected tree.
    """

    def plan(
        self,
        asset_pack: PlanningAssetPack,
        base_table: str,
        required_tables: Iterable[str],
        relationship_ref_ids: Optional[Iterable[str]] = None,
        usage: JoinUsage = "detail",
    ) -> JoinPlanningResult:
        if usage not in {"detail", "filter_scope", "aggregate"}:
            return _failure(
                "JOIN_USAGE_INVALID",
                "join usage must be one of detail, filter_scope, or aggregate",
                evidence=str(usage),
            )

        required = tuple(sorted({str(table) for table in required_tables if str(table)}))
        terminals = {base_table, *required}
        known_tables = set(asset_pack.known_tables())
        missing_tables = tuple(sorted(table for table in terminals if table not in known_tables))
        if missing_tables:
            return _failure(
                "JOIN_TABLE_NOT_GOVERNED",
                "all join tables must be present in the current PlanningAssetPack",
                tables=missing_tables,
                evidence=",".join(missing_tables),
            )

        compiled, invalid_by_ref, duplicate_gaps = _compile_relationships(asset_pack)
        if duplicate_gaps:
            return JoinPlanningResult(gaps=tuple(duplicate_gaps))

        explicit_refs = None if relationship_ref_ids is None else tuple(relationship_ref_ids)
        if explicit_refs is not None:
            duplicate_inputs = tuple(sorted({ref for ref in explicit_refs if explicit_refs.count(ref) > 1}))
            if duplicate_inputs:
                return _failure(
                    "JOIN_RELATIONSHIP_REF_DUPLICATE",
                    "explicit relationship refs must be unique",
                    evidence=",".join(duplicate_inputs),
                )
            all_known_refs = set(compiled) | set(invalid_by_ref)
            unknown = tuple(ref for ref in explicit_refs if ref not in all_known_refs)
            if unknown:
                return _failure(
                    "JOIN_RELATIONSHIP_REF_UNKNOWN",
                    "relationship refs must exactly match a recalled governed relationship ref",
                    evidence=",".join(unknown),
                )
            invalid = [gap for ref in explicit_refs for gap in invalid_by_ref.get(ref, ())]
            if invalid:
                return JoinPlanningResult(gaps=tuple(invalid))
            edges = [compiled[ref] for ref in explicit_refs]
            scope_gap = _explicit_scope_gap(base_table, terminals, edges)
            if scope_gap is not None:
                return JoinPlanningResult(gaps=(scope_gap,))
        else:
            edges = list(compiled.values())

        if terminals == {base_table}:
            if edges and explicit_refs is not None:
                return _failure(
                    "JOIN_RELATIONSHIP_SCOPE_NON_MINIMAL",
                    "no relationship is needed when every required table is the base table",
                    evidence=",".join(edge.ref_id for edge in edges),
                )
            return JoinPlanningResult(
                plan=JoinPlan(
                    base_table=base_table,
                    required_tables=required,
                    usage=usage,
                    execution_strategy=_execution_strategy(usage),
                    steps=(),
                )
            )

        solutions = _minimum_connector_trees(base_table, terminals, edges)
        if not solutions:
            path_gaps: List[JoinPlanningGap] = []
            if explicit_refs is None:
                path_gaps.extend(gap for gaps in invalid_by_ref.values() for gap in gaps)
            path_gaps.append(
                JoinPlanningGap(
                    code="JOIN_PATH_NOT_FOUND",
                    reason="recalled governed relationships do not connect the base table to every required table",
                    tables=tuple(sorted(terminals)),
                )
            )
            return JoinPlanningResult(gaps=tuple(path_gaps))
        if len(solutions) > 1:
            evidence = " | ".join(",".join(sorted(solution)) for solution in solutions[:2])
            return _failure(
                "JOIN_PATH_AMBIGUOUS",
                "more than one governed relationship tree has the same minimum size",
                tables=tuple(sorted(terminals)),
                evidence=evidence,
            )

        selected_refs = solutions[0]
        if explicit_refs is not None and selected_refs != frozenset(explicit_refs):
            unused = tuple(ref for ref in explicit_refs if ref not in selected_refs)
            return _failure(
                "JOIN_RELATIONSHIP_SCOPE_NON_MINIMAL",
                "every explicit relationship ref must be necessary for the minimum connector tree",
                evidence=",".join(unused),
            )

        selected_edges = [compiled[ref] for ref in selected_refs]
        steps = _ordered_steps(base_table, selected_edges, usage)
        if usage == "detail":
            unsafe = [step for step in steps if step.fanout_safe is not True]
            if unsafe:
                return JoinPlanningResult(
                    gaps=tuple(
                        JoinPlanningGap(
                            code="DETAIL_JOIN_GRAIN_UNPROVEN",
                            reason=(
                                "detail joins require one_to_one cardinality or an explicit "
                                "directional rowIdentityPreserved/resultGrain proof"
                            ),
                            relationship_ref_id=step.relationship_ref_id,
                            tables=(step.from_table, step.to_table),
                            evidence=step.safety_evidence,
                        )
                        for step in unsafe
                    )
                )
        if usage == "aggregate":
            unsafe = [step for step in steps if step.fanout_safe is not True]
            if unsafe:
                return JoinPlanningResult(
                    gaps=tuple(
                        JoinPlanningGap(
                            code="JOIN_FANOUT_UNPROVEN",
                            reason="aggregate joins require explicit directional cardinality or fanoutSafe metadata",
                            relationship_ref_id=step.relationship_ref_id,
                            tables=(step.from_table, step.to_table),
                            evidence=step.safety_evidence,
                        )
                        for step in unsafe
                    )
                )
            row_losing = [step for step in steps if step.base_rows_preserved is not True]
            if row_losing:
                return JoinPlanningResult(
                    gaps=tuple(
                        JoinPlanningGap(
                            code="JOIN_BASE_ROW_PRESERVATION_UNPROVEN",
                            reason=(
                                "aggregate joins must explicitly preserve base rows through joinType, "
                                "baseRowPreserved, or directional referentialCompleteness metadata"
                            ),
                            relationship_ref_id=step.relationship_ref_id,
                            tables=(step.from_table, step.to_table),
                            evidence=step.safety_evidence,
                        )
                        for step in row_losing
                    )
                )

        return JoinPlanningResult(
            plan=JoinPlan(
                base_table=base_table,
                required_tables=required,
                usage=usage,
                execution_strategy=_execution_strategy(usage),
                steps=tuple(steps),
            )
        )


def plan_governed_joins(
    asset_pack: PlanningAssetPack,
    base_table: str,
    required_tables: Iterable[str],
    relationship_ref_ids: Optional[Iterable[str]] = None,
    usage: JoinUsage = "detail",
) -> JoinPlanningResult:
    return GovernedJoinPlanner().plan(
        asset_pack=asset_pack,
        base_table=base_table,
        required_tables=required_tables,
        relationship_ref_ids=relationship_ref_ids,
        usage=usage,
    )


def _failure(
    code: str,
    reason: str,
    relationship_ref_id: str = "",
    tables: Tuple[str, ...] = (),
    evidence: str = "",
) -> JoinPlanningResult:
    return JoinPlanningResult(
        gaps=(
            JoinPlanningGap(
                code=code,
                reason=reason,
                relationship_ref_id=relationship_ref_id,
                tables=tables,
                evidence=evidence,
            ),
        )
    )


def _relationship_ref(relationship: RelationshipEntry) -> str:
    # source_ref_id is the governed identity. relationship_id is retained only as
    # a compatibility identity for old, already-recalled packs without source refs.
    return relationship.source_ref_id or relationship.relationship_id


def _compile_relationships(
    asset_pack: PlanningAssetPack,
) -> Tuple[Dict[str, _GovernedEdge], Dict[str, Tuple[JoinPlanningGap, ...]], List[JoinPlanningGap]]:
    compiled: Dict[str, _GovernedEdge] = {}
    invalid: Dict[str, Tuple[JoinPlanningGap, ...]] = {}
    duplicate_gaps: List[JoinPlanningGap] = []
    seen_refs: Set[str] = set()

    for relationship in asset_pack.relationships:
        ref_id = _relationship_ref(relationship)
        if not ref_id:
            continue
        if ref_id in seen_refs:
            compiled.pop(ref_id, None)
            invalid.pop(ref_id, None)
            duplicate_gaps.append(
                JoinPlanningGap(
                    code="JOIN_RELATIONSHIP_REF_DUPLICATE",
                    reason="governed relationship refs must identify exactly one relationship",
                    relationship_ref_id=ref_id,
                )
            )
            continue
        seen_refs.add(ref_id)
        gaps = _relationship_gaps(asset_pack, relationship, ref_id)
        if gaps:
            invalid[ref_id] = tuple(gaps)
            continue
        left_columns = tuple(str(pair["leftColumn"]) for pair in relationship.join_keys)
        right_columns = tuple(str(pair["rightColumn"]) for pair in relationship.join_keys)
        compiled[ref_id] = _GovernedEdge(
            ref_id=ref_id,
            relationship=relationship,
            left_columns=left_columns,
            right_columns=right_columns,
        )
    return compiled, invalid, duplicate_gaps


def _relationship_gaps(
    asset_pack: PlanningAssetPack,
    relationship: RelationshipEntry,
    ref_id: str,
) -> List[JoinPlanningGap]:
    gaps: List[JoinPlanningGap] = []
    endpoints = (relationship.left_table, relationship.right_table)
    known_tables = set(asset_pack.known_tables())
    if not all(endpoints) or relationship.left_table == relationship.right_table:
        gaps.append(
            JoinPlanningGap(
                code="JOIN_RELATIONSHIP_INVALID",
                reason="a relationship must connect two different non-empty tables",
                relationship_ref_id=ref_id,
                tables=endpoints,
            )
        )
        return gaps
    unknown_tables = tuple(table for table in endpoints if table not in known_tables)
    if unknown_tables:
        gaps.append(
            JoinPlanningGap(
                code="JOIN_RELATIONSHIP_TABLE_UNKNOWN",
                reason="relationship endpoints must be present in the current PlanningAssetPack",
                relationship_ref_id=ref_id,
                tables=unknown_tables,
            )
        )
        return gaps
    if not relationship.join_keys:
        gaps.append(
            JoinPlanningGap(
                code="JOIN_KEYS_MISSING",
                reason="a governed relationship must declare at least one complete join-key pair",
                relationship_ref_id=ref_id,
                tables=endpoints,
            )
        )
        return gaps

    left_columns = set(asset_pack.known_columns(relationship.left_table))
    right_columns = set(asset_pack.known_columns(relationship.right_table))
    for pair in relationship.join_keys:
        left = str(pair.get("leftColumn") or "")
        right = str(pair.get("rightColumn") or "")
        if not left or not right:
            gaps.append(
                JoinPlanningGap(
                    code="JOIN_KEYS_MISSING",
                    reason="every governed join-key pair must declare both leftColumn and rightColumn",
                    relationship_ref_id=ref_id,
                    tables=endpoints,
                    evidence=json.dumps(pair, ensure_ascii=False, sort_keys=True),
                )
            )
            continue
        if left not in left_columns:
            gaps.append(
                JoinPlanningGap(
                    code="JOIN_KEY_FIELD_NOT_IN_TABLE",
                    reason="left join-key field is not declared on the relationship's left table",
                    relationship_ref_id=ref_id,
                    tables=(relationship.left_table,),
                    evidence=left,
                )
            )
        if right not in right_columns:
            gaps.append(
                JoinPlanningGap(
                    code="JOIN_KEY_FIELD_NOT_IN_TABLE",
                    reason="right join-key field is not declared on the relationship's right table",
                    relationship_ref_id=ref_id,
                    tables=(relationship.right_table,),
                    evidence=right,
                )
            )
        if left in left_columns and right in right_columns:
            left_types = _column_type_families(asset_pack, relationship.left_table, left)
            right_types = _column_type_families(asset_pack, relationship.right_table, right)
            if len(left_types) != 1 or len(right_types) != 1:
                gaps.append(
                    JoinPlanningGap(
                        code="JOIN_KEY_TYPE_UNKNOWN",
                        reason=(
                            "every executable join-key side must have exactly one governed data-type family"
                        ),
                        relationship_ref_id=ref_id,
                        tables=endpoints,
                        evidence=(
                            "%s.%s=%s,%s.%s=%s"
                            % (
                                relationship.left_table,
                                left,
                                "/".join(sorted(left_types)) or "unknown",
                                relationship.right_table,
                                right,
                                "/".join(sorted(right_types)) or "unknown",
                            )
                        ),
                    )
                )
            elif left_types != right_types:
                gaps.append(
                    JoinPlanningGap(
                        code="JOIN_KEY_TYPE_MISMATCH",
                        reason="join-key data types must belong to the same governed compatibility family",
                        relationship_ref_id=ref_id,
                        tables=endpoints,
                        evidence=(
                            "%s.%s=%s,%s.%s=%s"
                            % (
                                relationship.left_table,
                                left,
                                next(iter(left_types)),
                                relationship.right_table,
                                right,
                                next(iter(right_types)),
                            )
                        ),
                    )
                )

    if not any(
        gap.code in {"JOIN_KEYS_MISSING", "JOIN_KEY_FIELD_NOT_IN_TABLE"}
        for gap in gaps
    ):
        tenant_gap = _tenant_isolation_gap(asset_pack, relationship, ref_id)
        if tenant_gap is not None:
            gaps.append(tenant_gap)
    return gaps


_TYPE_FAMILIES: Dict[str, str] = {
    "tinyint": "numeric",
    "smallint": "numeric",
    "int": "numeric",
    "integer": "numeric",
    "bigint": "numeric",
    "decimal": "numeric",
    "numeric": "numeric",
    "number": "numeric",
    "float": "numeric",
    "double": "numeric",
    "real": "numeric",
    "string": "string",
    "varchar": "string",
    "char": "string",
    "text": "string",
    "boolean": "boolean",
    "bool": "boolean",
    "date": "date",
    "timestamp": "datetime",
    "datetime": "datetime",
    "time": "time",
    "binary": "binary",
    "varbinary": "binary",
    "uuid": "uuid",
}


def _column_type_families(
    asset_pack: PlanningAssetPack,
    table: str,
    column: str,
) -> Set[str]:
    raw_types: List[Any] = []
    entries = [
        entry
        for entry in (*asset_pack.fields, *asset_pack.entity_keys)
        if entry.table == table and (entry.key == column or column in entry.columns)
    ]
    for entry in entries:
        raw_types.extend(_metadata_values(entry.metadata, "datatype"))
        raw_types.extend(_metadata_values(entry.metadata, "physicaltype"))

    for table_entry in (entry for entry in asset_pack.tables if entry.table == table):
        for descriptor in _column_descriptors(table_entry.metadata, column):
            raw_types.extend(_direct_metadata_values(descriptor, {"datatype", "physicaltype", "type"}))
        for value in _metadata_values(table_entry.metadata, "columntypes"):
            if isinstance(value, Mapping) and column in value:
                raw_types.append(value[column])

    return {
        family
        for raw_type in raw_types
        if (family := _normalize_type_family(raw_type))
    }


def _column_descriptors(metadata: Mapping[str, Any], column: str) -> List[Mapping[str, Any]]:
    descriptors: List[Mapping[str, Any]] = []
    for candidate in _walk_mappings(metadata):
        identities = _direct_metadata_values(
            candidate,
            {"column", "columnname", "field", "fieldname", "physicalcolumn", "name"},
        )
        if any(str(identity) == column for identity in identities):
            descriptors.append(candidate)
    return descriptors


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _direct_metadata_values(payload: Mapping[str, Any], keys: Set[str]) -> List[Any]:
    return [value for key, value in payload.items() if _normalize_metadata_key(key) in keys]


def _normalize_type_family(value: Any) -> str:
    if isinstance(value, Mapping):
        nested = _direct_metadata_values(value, {"datatype", "physicaltype", "type"})
        return _normalize_type_family(nested[0]) if len(nested) == 1 else ""
    normalized = str(value or "").strip().casefold()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"\(.*\)$", "", normalized)
    if normalized.startswith("decimal"):
        normalized = "decimal"
    elif normalized.startswith("varchar"):
        normalized = "varchar"
    elif normalized.startswith("char"):
        normalized = "char"
    elif normalized.startswith("timestamp"):
        normalized = "timestamp"
    return _TYPE_FAMILIES.get(normalized, "")


_TENANT_ROLES = {
    "tenant",
    "tenantkey",
    "tenantscope",
    "tenantscopekey",
    "rowscopekey",
    "rowaccesskey",
    "securityscopekey",
}
_GLOBAL_SCOPE_TYPES = {"global", "public", "shared", "unscoped", "none"}


def _tenant_isolation_gap(
    asset_pack: PlanningAssetPack,
    relationship: RelationshipEntry,
    ref_id: str,
) -> Optional[JoinPlanningGap]:
    left_scope, left_scope_columns, left_evidence = _table_scope_contract(
        asset_pack,
        relationship.left_table,
    )
    right_scope, right_scope_columns, right_evidence = _table_scope_contract(
        asset_pack,
        relationship.right_table,
    )
    if "tenant" not in {left_scope, right_scope}:
        return None
    if {left_scope, right_scope} == {"tenant", "global"}:
        return None

    metadata = _relationship_metadata(relationship)
    declared_safe, declaration_evidence = _declared_tenant_composite_safety(metadata)
    actual_pairs = {
        (str(pair.get("leftColumn") or ""), str(pair.get("rightColumn") or ""))
        for pair in relationship.join_keys
    }
    tenant_pairs = {
        pair
        for pair in actual_pairs
        if pair[0] in left_scope_columns and pair[1] in right_scope_columns
    }
    all_scope_columns_covered = bool(left_scope_columns and right_scope_columns) and (
        left_scope_columns <= {pair[0] for pair in tenant_pairs}
        and right_scope_columns <= {pair[1] for pair in tenant_pairs}
    )
    inferred_composite_safe = (
        left_scope == right_scope == "tenant"
        and all_scope_columns_covered
        and len(actual_pairs) > len(tenant_pairs)
    )
    declared_composite_safe = declared_safe is True and len(actual_pairs) >= 2
    if declared_safe is not False and (inferred_composite_safe or declared_composite_safe):
        return None

    evidence = ";".join(
        item
        for item in (
            "leftScope=%s(%s)" % (left_scope, left_evidence),
            "rightScope=%s(%s)" % (right_scope, right_evidence),
            declaration_evidence,
            "scopeKeyPairs=%d,totalKeyPairs=%d" % (len(tenant_pairs), len(actual_pairs)),
        )
        if item
    )
    return JoinPlanningGap(
        code="JOIN_TENANT_ISOLATION_UNPROVEN",
        reason=(
            "tenant-scoped relationships require a governed tenant-safe composite join key "
            "or an explicitly global opposite endpoint"
        ),
        relationship_ref_id=ref_id,
        tables=(relationship.left_table, relationship.right_table),
        evidence=evidence,
    )


def _table_scope_contract(
    asset_pack: PlanningAssetPack,
    table: str,
) -> Tuple[str, Set[str], str]:
    tenant_columns: Set[str] = set()
    scope_candidates: Set[str] = set()
    evidence: List[str] = []
    table_entries = [entry for entry in asset_pack.tables if entry.table == table]
    for entry in table_entries:
        for policy in _metadata_values(entry.metadata, "rowaccesspolicy"):
            if not isinstance(policy, Mapping):
                continue
            scope_type_values = _direct_metadata_values(policy, {"scopetype", "scope"})
            scope_type = _normalize_metadata_key(scope_type_values[0]) if scope_type_values else ""
            columns = _scope_columns_from_policy(policy)
            tenant_columns.update(columns)
            if scope_type in _GLOBAL_SCOPE_TYPES:
                scope_candidates.add("global")
            elif scope_type or columns:
                scope_candidates.add("tenant")
            evidence.append("rowAccessPolicy:%s" % (scope_type or "scoped"))

        for value in _metadata_values(entry.metadata, "tenantscoped"):
            parsed = _strict_bool(value)
            if parsed is True:
                scope_candidates.add("tenant")
                evidence.append("tenantScoped:true")
            elif parsed is False:
                scope_candidates.add("global")
                evidence.append("tenantScoped:false")

        for descriptor in _walk_mappings(entry.metadata):
            roles = _direct_metadata_values(descriptor, {"semanticrole", "role"})
            normalized_roles = {_normalize_metadata_key(role) for role in roles}
            if not (normalized_roles & _TENANT_ROLES):
                continue
            identities = _direct_metadata_values(
                descriptor,
                {"column", "columnname", "field", "fieldname", "physicalcolumn", "name"},
            )
            tenant_columns.update(str(identity) for identity in identities if str(identity))
            scope_candidates.add("tenant")
            evidence.append("semanticRole:tenant_scope")

    for field in (*asset_pack.fields, *asset_pack.entity_keys):
        if field.table != table:
            continue
        roles = {
            _normalize_metadata_key(value)
            for key in ("semanticrole", "role")
            for value in _metadata_values(field.metadata, key)
        }
        if not (roles & _TENANT_ROLES):
            continue
        tenant_columns.update(field.columns or ([field.key] if field.key else []))
        scope_candidates.add("tenant")
        evidence.append("semanticRole:tenant_scope")

    if "tenant" in scope_candidates:
        return "tenant", tenant_columns, ",".join(sorted(set(evidence)))
    if scope_candidates == {"global"}:
        return "global", set(), ",".join(sorted(set(evidence)))
    return "undeclared", set(), "no governed scope declaration"


def _scope_columns_from_policy(policy: Mapping[str, Any]) -> Set[str]:
    columns: Set[str] = set()
    for key, value in policy.items():
        if _normalize_metadata_key(key) not in {
            "filtercolumn",
            "filtercolumns",
            "scopecolumn",
            "scopecolumns",
        }:
            continue
        if isinstance(value, str) and value:
            columns.add(value)
        elif isinstance(value, (list, tuple, set)):
            columns.update(str(item) for item in value if str(item))
    return columns


def _declared_tenant_composite_safety(
    metadata: Mapping[str, Any],
) -> Tuple[Optional[bool], str]:
    values: List[Any] = []
    for key in (
        "tenantsafecompositekey",
        "tenantcompositekeysafe",
        "tenantisolationsafe",
        "tenantsafe",
    ):
        values.extend(_metadata_values(metadata, key))
    for declaration in _metadata_values(metadata, "tenantsafety"):
        if not isinstance(declaration, Mapping):
            values.append(declaration)
            continue
        safe_values = _direct_metadata_values(
            declaration,
            {"safe", "tenantsafe", "compositekeysafe", "tenantisolationsafe"},
        )
        composite_values = _direct_metadata_values(
            declaration,
            {"composite", "compositekey", "compositekeysafe"},
        )
        if safe_values and composite_values:
            safe = {_strict_bool(value) for value in safe_values}
            composite = {_strict_bool(value) for value in composite_values}
            if safe == {True} and composite == {True}:
                values.append(True)
            else:
                values.append(False)

    if not values:
        return None, "no tenant-safe composite-key declaration"
    parsed = [_strict_bool(value) for value in values]
    candidates = {value for value in parsed if value is not None}
    if any(value is None for value in parsed) or len(candidates) != 1:
        return False, "tenant-safe composite-key declaration is malformed or conflicting"
    value = next(iter(candidates))
    return value, "tenant-safe composite-key declaration=%s" % str(value).lower()


def _explicit_scope_gap(
    base_table: str,
    terminals: Set[str],
    edges: Sequence[_GovernedEdge],
) -> Optional[JoinPlanningGap]:
    parent: Dict[str, str] = {}

    def root(table: str) -> str:
        parent.setdefault(table, table)
        while parent[table] != table:
            parent[table] = parent[parent[table]]
            table = parent[table]
        return table

    for edge in edges:
        left_root = root(edge.left_table)
        right_root = root(edge.right_table)
        if left_root == right_root:
            return JoinPlanningGap(
                code="JOIN_RELATIONSHIP_SCOPE_CYCLIC",
                reason="explicit relationship refs must form an acyclic tree",
                relationship_ref_id=edge.ref_id,
                tables=(edge.left_table, edge.right_table),
            )
        parent[left_root] = right_root

    if terminals != {base_table} and not edges:
        return JoinPlanningGap(
            code="JOIN_RELATIONSHIP_SCOPE_DISCONNECTED",
            reason="explicit relationship refs do not connect the required tables",
            tables=tuple(sorted(terminals)),
        )
    base_root = root(base_table)
    edge_tables = {table for edge in edges for table in (edge.left_table, edge.right_table)}
    disconnected = tuple(sorted(table for table in terminals | edge_tables if root(table) != base_root))
    if disconnected:
        return JoinPlanningGap(
            code="JOIN_RELATIONSHIP_SCOPE_DISCONNECTED",
            reason="every explicit relationship must belong to the base table's connected component",
            tables=disconnected,
        )
    return None


def _minimum_connector_trees(
    base_table: str,
    terminals: Set[str],
    edges: Sequence[_GovernedEdge],
) -> List[FrozenSet[str]]:
    """Enumerate rooted trees by edge count and stop at the first complete level."""

    edge_by_ref = {edge.ref_id: edge for edge in edges}
    frontier: Set[FrozenSet[str]] = {frozenset()}
    for _depth in range(len(edges) + 1):
        solutions = [state for state in frontier if terminals <= _tables_for_tree(base_table, state, edge_by_ref)]
        if solutions:
            return sorted(solutions, key=lambda state: tuple(sorted(state)))[:2]
        next_frontier: Set[FrozenSet[str]] = set()
        for state in frontier:
            connected = _tables_for_tree(base_table, state, edge_by_ref)
            for edge in edges:
                if edge.ref_id in state:
                    continue
                left_connected = edge.left_table in connected
                right_connected = edge.right_table in connected
                if left_connected == right_connected:
                    continue
                next_frontier.add(state | {edge.ref_id})
        if not next_frontier:
            return []
        frontier = next_frontier
    return []


def _tables_for_tree(
    base_table: str,
    refs: FrozenSet[str],
    edge_by_ref: Mapping[str, _GovernedEdge],
) -> Set[str]:
    tables = {base_table}
    for ref_id in refs:
        edge = edge_by_ref[ref_id]
        tables.update((edge.left_table, edge.right_table))
    return tables


def _ordered_steps(base_table: str, edges: Sequence[_GovernedEdge], usage: JoinUsage) -> List[JoinStep]:
    pending = {edge.ref_id: edge for edge in edges}
    visited = {base_table}
    steps: List[JoinStep] = []
    while pending:
        candidates: List[Tuple[str, str, _GovernedEdge]] = []
        for edge in pending.values():
            if (edge.left_table in visited) == (edge.right_table in visited):
                continue
            from_table = edge.left_table if edge.left_table in visited else edge.right_table
            to_table = edge.right_table if from_table == edge.left_table else edge.left_table
            candidates.append((from_table, to_table, edge))
        if not candidates:
            # _minimum_connector_trees only returns connected trees, so this is a
            # defensive guard rather than a recoverable planning branch.
            raise ValueError("selected governed relationship tree is not rooted at base_table")
        for from_table, to_table, edge in sorted(candidates, key=lambda item: (item[0], item[1], item[2].ref_id)):
            if edge.ref_id not in pending or from_table not in visited or to_table in visited:
                continue
            forward = from_table == edge.left_table
            from_columns = edge.left_columns if forward else edge.right_columns
            to_columns = edge.right_columns if forward else edge.left_columns
            (
                cardinality,
                fanout_safe,
                join_type,
                base_rows_preserved,
                safety_evidence,
            ) = _step_safety(edge.relationship, forward, usage)
            steps.append(
                JoinStep(
                    relationship_ref_id=edge.ref_id,
                    relationship_id=edge.relationship.relationship_id,
                    from_table=from_table,
                    to_table=to_table,
                    keys=tuple(
                        JoinKeyBinding(from_column=from_column, to_column=to_column)
                        for from_column, to_column in zip(from_columns, to_columns)
                    ),
                    strategy="exists" if usage == "filter_scope" else "relationship_join",
                    cardinality=cardinality,
                    fanout_safe=fanout_safe,
                    join_type=join_type,
                    base_rows_preserved=base_rows_preserved,
                    safety_evidence=safety_evidence,
                )
            )
            visited.add(to_table)
            pending.pop(edge.ref_id)
    return steps


def _execution_strategy(usage: JoinUsage) -> str:
    return "semi_join" if usage == "filter_scope" else "relationship_join"


def _step_safety(
    relationship: RelationshipEntry,
    forward: bool,
    usage: JoinUsage,
) -> Tuple[str, Optional[bool], str, Optional[bool], str]:
    if usage == "filter_scope":
        return "", True, "exists", None, "EXISTS/semi-join cannot duplicate base rows"
    if usage == "detail":
        cardinality, grain_safe, grain_evidence = _detail_grain_safety(relationship, forward)
        return cardinality, grain_safe, _declared_join_type(_relationship_metadata(relationship)), None, grain_evidence
    cardinality, fanout_safe, fanout_evidence = _aggregate_fanout_safety(relationship, forward)
    join_type, base_rows_preserved, row_evidence = _aggregate_base_row_safety(relationship, forward)
    return (
        cardinality,
        fanout_safe,
        join_type,
        base_rows_preserved,
        "%s;%s" % (fanout_evidence, row_evidence),
    )


def _detail_grain_safety(
    relationship: RelationshipEntry,
    forward: bool,
) -> Tuple[str, Optional[bool], str]:
    metadata = _relationship_metadata(relationship)
    cardinality = _declared_cardinality(metadata)
    if cardinality == "one_to_one":
        return cardinality, True, "cardinality=one_to_one"

    for generic_key, directional_stem in (
        ("rowidentitypreserved", "rowidentitypreserved"),
        ("resultgrainpreserved", "resultgrainpreserved"),
    ):
        declared, value, evidence = _directional_bool_contract(
            metadata,
            generic_key,
            directional_stem,
            forward,
        )
        if declared:
            return cardinality, value, evidence

    grain_declared, grain_safe, grain_evidence = _directional_result_grain(
        metadata,
        relationship,
        forward,
    )
    if grain_declared:
        return cardinality, grain_safe, grain_evidence
    return (
        cardinality,
        None,
        "cardinality=%s;no directional rowIdentityPreserved/resultGrain proof"
        % (cardinality or "undeclared"),
    )


def _aggregate_base_row_safety(
    relationship: RelationshipEntry,
    forward: bool,
) -> Tuple[str, Optional[bool], str]:
    metadata = _relationship_metadata(relationship)
    join_type = _declared_join_type(metadata)
    preserved_declared, preserved, preserved_evidence = _directional_bool_contract(
        metadata,
        "baserowpreserved",
        "baserowpreserved",
        forward,
        plural_generic_key="baserowspreserved",
        plural_directional_stem="baserowspreserved",
    )
    if preserved_declared:
        return join_type, preserved, preserved_evidence

    completeness_declared, complete, completeness_evidence = _directional_bool_contract(
        metadata,
        "referentialcompleteness",
        "referentiallycomplete",
        forward,
    )
    if join_type == "full_outer":
        return join_type, True, "joinType=full_outer preserves rows from both endpoints"
    if join_type == "left_outer":
        safe = forward
        return join_type, safe, "joinType=left_outer,direction=%s" % _direction_name(forward)
    if join_type == "right_outer":
        safe = not forward
        return join_type, safe, "joinType=right_outer,direction=%s" % _direction_name(forward)
    if complete is True:
        return join_type, True, completeness_evidence
    if completeness_declared:
        return join_type, False, completeness_evidence
    if join_type == "inner":
        return join_type, None, "joinType=inner;referential completeness is not proven"
    if join_type:
        return join_type, None, "joinType=%s does not prove base-row preservation" % join_type
    return "", None, "joinType/baseRowPreserved/referentialCompleteness are undeclared"


def _declared_join_type(metadata: Mapping[str, Any]) -> str:
    candidates = {
        normalized
        for value in (
            *_metadata_values(metadata, "jointype"),
            *_metadata_values(metadata, "sqljointype"),
        )
        if (normalized := _normalize_join_type(value))
    }
    return next(iter(candidates)) if len(candidates) == 1 else ""


def _normalize_join_type(value: Any) -> str:
    normalized = _normalize_metadata_key(value)
    aliases = {
        "inner": "inner",
        "innerjoin": "inner",
        "left": "left_outer",
        "leftjoin": "left_outer",
        "leftouter": "left_outer",
        "leftouterjoin": "left_outer",
        "right": "right_outer",
        "rightjoin": "right_outer",
        "rightouter": "right_outer",
        "rightouterjoin": "right_outer",
        "full": "full_outer",
        "fulljoin": "full_outer",
        "fullouter": "full_outer",
        "fullouterjoin": "full_outer",
    }
    return aliases.get(normalized, "")


def _directional_bool_contract(
    metadata: Mapping[str, Any],
    generic_key: str,
    directional_stem: str,
    forward: bool,
    *,
    plural_generic_key: str = "",
    plural_directional_stem: str = "",
) -> Tuple[bool, Optional[bool], str]:
    direction_prefix = "lefttoright" if forward else "righttoleft"
    keys = {direction_prefix + directional_stem, direction_prefix + generic_key}
    if plural_directional_stem:
        keys.add(direction_prefix + plural_directional_stem)
    applicable: List[Any] = []
    for key in keys:
        applicable.extend(_metadata_values(metadata, key))

    generic_keys = {generic_key}
    if plural_generic_key:
        generic_keys.add(plural_generic_key)
    accepted_map_keys = {"lefttoright", "forward"} if forward else {"righttoleft", "reverse"}
    for key in generic_keys:
        for value in _metadata_values(metadata, key):
            parsed = _strict_bool(value)
            if parsed is not None or not isinstance(value, Mapping):
                applicable.append(value)
                continue
            for map_key, directional_value in value.items():
                if _normalize_metadata_key(map_key) in accepted_map_keys:
                    applicable.append(directional_value)

    label = plural_generic_key or generic_key
    direction = _direction_name(forward)
    if not applicable:
        return False, None, ""
    parsed_values = [_strict_bool(value) for value in applicable]
    candidates = {value for value in parsed_values if value is not None}
    if any(value is None for value in parsed_values) or len(candidates) != 1:
        return True, None, "%s declaration is malformed or conflicting,direction=%s" % (label, direction)
    value = next(iter(candidates))
    return True, value, "%s=%s,direction=%s" % (label, str(value).lower(), direction)


def _directional_result_grain(
    metadata: Mapping[str, Any],
    relationship: RelationshipEntry,
    forward: bool,
) -> Tuple[bool, Optional[bool], str]:
    values = _metadata_values(metadata, "resultgrain")
    if not values:
        return False, None, ""
    expected_side = "left" if forward else "right"
    expected_table = relationship.left_table if forward else relationship.right_table
    candidates: Set[bool] = set()
    malformed = False
    for value in values:
        if isinstance(value, Mapping):
            owner_values = _direct_metadata_values(
                value,
                {"owner", "side", "table", "basetable", "grainowner"},
            )
            preserved_values = _direct_metadata_values(value, {"preserved", "preservesbase"})
            if preserved_values:
                parsed = {_strict_bool(item) for item in preserved_values}
                if parsed == {True}:
                    candidates.add(True)
                elif parsed == {False}:
                    candidates.add(False)
                else:
                    malformed = True
            value_candidates = owner_values
        else:
            value_candidates = [value]
        for candidate in value_candidates:
            normalized = _normalize_metadata_key(candidate)
            if normalized in {"base", "basetable", "from", "fromtable"}:
                candidates.add(True)
            elif normalized == expected_side or normalized == _normalize_metadata_key(expected_table):
                candidates.add(True)
            elif normalized in {"left", "right"}:
                candidates.add(False)
            elif not isinstance(candidate, bool):
                malformed = True
    if malformed or len(candidates) != 1:
        return True, None, "resultGrain declaration is malformed or conflicting,direction=%s" % _direction_name(forward)
    safe = next(iter(candidates))
    return True, safe, "resultGrain preserves base=%s,direction=%s" % (str(safe).lower(), _direction_name(forward))


def _direction_name(forward: bool) -> str:
    return "left_to_right" if forward else "right_to_left"


def _aggregate_fanout_safety(
    relationship: RelationshipEntry,
    forward: bool,
) -> Tuple[str, Optional[bool], str]:
    metadata = _relationship_metadata(relationship)
    fanout_declared, explicit_fanout, fanout_evidence = _directional_fanout_declaration(metadata, forward)
    if fanout_declared:
        return "", explicit_fanout, fanout_evidence

    cardinality = _declared_cardinality(metadata)
    if not cardinality:
        return "", None, "relationship has no recognized cardinality or fanoutSafe declaration"
    safe_forward = cardinality in {"one_to_one", "many_to_one"}
    safe_reverse = cardinality in {"one_to_one", "one_to_many"}
    safe = safe_forward if forward else safe_reverse
    direction = "left_to_right" if forward else "right_to_left"
    return cardinality, safe, "cardinality=%s,direction=%s" % (cardinality, direction)


def _relationship_metadata(relationship: RelationshipEntry) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    dynamic_metadata = getattr(relationship, "metadata", None)
    if isinstance(dynamic_metadata, Mapping):
        payload["metadata"] = dict(dynamic_metadata)
    description = relationship.description.strip()
    if description:
        try:
            decoded = json.loads(description)
        except (TypeError, ValueError):
            decoded = None
        if isinstance(decoded, Mapping):
            payload["description"] = dict(decoded)
    return payload


def _metadata_values(payload: Mapping[str, Any], normalized_key: str) -> List[Any]:
    values: List[Any] = []
    for key, value in payload.items():
        if _normalize_metadata_key(key) == normalized_key:
            values.append(value)
        if isinstance(value, Mapping):
            values.extend(_metadata_values(value, normalized_key))
    return values


def _normalize_metadata_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _directional_fanout_declaration(
    metadata: Mapping[str, Any],
    forward: bool,
) -> Tuple[bool, Optional[bool], str]:
    direction_key = "lefttorightfanoutsafe" if forward else "righttoleftfanoutsafe"
    direct_values = _metadata_values(metadata, direction_key)
    applicable_values: List[Any] = list(direct_values)
    malformed_directional_map = False
    candidates: Set[bool] = set()
    for value in _metadata_values(metadata, "fanoutsafe"):
        parsed = _strict_bool(value)
        if parsed is not None:
            applicable_values.append(value)
            continue
        if not isinstance(value, Mapping):
            applicable_values.append(value)
            continue
        accepted_keys = {"lefttoright", "forward"} if forward else {"righttoleft", "reverse"}
        matched = False
        for key, directional_value in value.items():
            if _normalize_metadata_key(key) in accepted_keys:
                applicable_values.append(directional_value)
                matched = True
        if matched and any(_strict_bool(item) is None for item in applicable_values):
            malformed_directional_map = True

    if not applicable_values:
        return False, None, ""
    parsed_values = [_strict_bool(value) for value in applicable_values]
    candidates.update(value for value in parsed_values if value is not None)
    direction = "left_to_right" if forward else "right_to_left"
    if malformed_directional_map or any(value is None for value in parsed_values) or len(candidates) != 1:
        return True, None, "fanoutSafe declaration is malformed or conflicting for direction=%s" % direction
    value = next(iter(candidates))
    return True, value, "fanoutSafe=%s,direction=%s" % (str(value).lower(), direction)


def _strict_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.casefold() in {"true", "false"}:
        return value.casefold() == "true"
    return None


def _declared_cardinality(metadata: Mapping[str, Any]) -> str:
    candidates: Set[str] = set()
    for value in _metadata_values(metadata, "cardinality"):
        normalized = _normalize_cardinality(value)
        if normalized:
            candidates.add(normalized)
    if len(candidates) == 1:
        return next(iter(candidates))
    return ""


def _normalize_cardinality(value: Any) -> str:
    if isinstance(value, Mapping):
        sides = {_normalize_metadata_key(key): _normalize_metadata_key(item) for key, item in value.items()}
        left = sides.get("left") or sides.get("leftside")
        right = sides.get("right") or sides.get("rightside")
        if left in {"one", "1"} and right in {"one", "1"}:
            return "one_to_one"
        if left in {"many", "m", "n"} and right in {"one", "1"}:
            return "many_to_one"
        if left in {"one", "1"} and right in {"many", "m", "n"}:
            return "one_to_many"
        if left in {"many", "m", "n"} and right in {"many", "m", "n"}:
            return "many_to_many"
        return ""
    normalized = _normalize_metadata_key(value)
    aliases = {
        "11": "one_to_one",
        "onetoone": "one_to_one",
        "m1": "many_to_one",
        "n1": "many_to_one",
        "manytoone": "many_to_one",
        "1m": "one_to_many",
        "1n": "one_to_many",
        "onetomany": "one_to_many",
        "mm": "many_to_many",
        "nn": "many_to_many",
        "manytomany": "many_to_many",
    }
    return aliases.get(normalized, "")
