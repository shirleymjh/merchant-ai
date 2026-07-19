from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope
from sqlglot.tokens import Tokenizer

from merchant_ai.models import NodeExecutionContext, PlanningAssetPack, QuestionIntent


def inclusive_day_interval(days: Any) -> int:
    try:
        value = int(days or 0)
    except (TypeError, ValueError):
        value = 0
    return max(value - 1, 0)


def normalize_inclusive_relative_window_sql(sql: str, days: Any) -> str:
    text = str(sql or "")
    try:
        requested_days = int(days or 0)
    except (TypeError, ValueError):
        requested_days = 0
    if requested_days <= 0:
        return text
    inclusive_interval = inclusive_day_interval(requested_days)
    replacements = _date_sub_replacements(
        text,
        expected_days=requested_days,
        replacement_days=inclusive_interval,
    )
    return _apply_text_replacements(text, replacements)


@dataclass(frozen=True)
class SplitDetailSqlChunk:
    """One required slice of a split detail-query window."""

    index: int
    offset_start_days: int
    offset_end_days: int
    start_date: str
    end_date: str
    sql: str

    def contract_payload(self) -> Dict[str, Any]:
        return {
            "chunkIndex": self.index,
            "offsetStartDays": self.offset_start_days,
            "offsetEndDays": self.offset_end_days,
            "startDate": self.start_date,
            "endDate": self.end_date,
        }


@dataclass(frozen=True)
class SplitDetailSqlPlan:
    """Bounded execution plan plus the full requested-window obligation."""

    requested_days: int
    requested_start_date: str
    requested_end_date: str
    anchor_date: str
    chunk_days: int
    max_chunks: int
    required_chunk_count: int
    chunks: Tuple[SplitDetailSqlChunk, ...]

    @property
    def truncated(self) -> bool:
        return len(self.chunks) < self.required_chunk_count

    @property
    def omitted_chunk_count(self) -> int:
        return max(0, self.required_chunk_count - len(self.chunks))


def split_detail_chunk_restriction_sql(
    time_column: str,
    anchor_expression: str,
    offset_start_days: int,
    offset_end_days: int,
) -> str:
    """Build the deterministic, half-open partition restriction for one chunk."""

    quoted_column = quote_identifier(normalize_identifier(time_column))
    lower_bound = "%s >= DATE_SUB(%s, INTERVAL %d DAY)" % (
        quoted_column,
        anchor_expression,
        max(0, int(offset_end_days or 0)),
    )
    if int(offset_start_days or 0) <= 0:
        upper_bound = "%s < DATE_ADD(%s, INTERVAL 1 DAY)" % (
            quoted_column,
            anchor_expression,
        )
    else:
        upper_bound = "%s < DATE_SUB(%s, INTERVAL %d DAY)" % (
            quoted_column,
            anchor_expression,
            max(0, int(offset_start_days or 0) - 1),
        )
    return "(%s AND %s)" % (lower_bound, upper_bound)


def build_split_detail_sql_plan(
    sql: str,
    days: int,
    chunk_days: int,
    max_chunks: int,
    limit: int,
    time_column: str,
    anchor_date: str = "",
) -> Optional[SplitDetailSqlPlan]:
    """Plan resource-safe SQL chunks without losing the original window size.

    ``max_chunks`` limits scheduled work, not the semantic obligation.  The
    returned plan therefore keeps ``required_chunk_count`` so callers can fail
    closed when the cap omits any portion of the requested window.
    """

    text = str(sql or "").strip()
    partition_column = normalize_identifier(time_column)
    parsed = parse_sql_for_binding(text)
    if (
        not text
        or parsed is None
        or parsed.find(exp.Group) is not None
        or parsed.find(exp.Union) is not None
        or parsed.find(exp.Join) is not None
    ):
        return None
    if not partition_column or not sql_references_column(text, partition_column):
        return None
    total_days = max(1, int(days or 0))
    window_days = max(1, int(chunk_days or 1))
    chunk_cap = max(1, int(max_chunks or 1))
    capped_limit = max(1, int(limit or 1))
    required_chunk_count = (total_days + window_days - 1) // window_days
    scheduled_chunk_count = min(required_chunk_count, chunk_cap)
    anchor_expr = "CURDATE()"
    parsed_anchor: Optional[date] = None
    if anchor_date:
        escaped_anchor = str(anchor_date).replace("'", "''")
        anchor_expr = "'%s'" % escaped_anchor
        try:
            parsed_anchor = date.fromisoformat(str(anchor_date))
        except ValueError:
            parsed_anchor = None
    requested_start_date = (
        (parsed_anchor - timedelta(days=total_days - 1)).isoformat()
        if parsed_anchor
        else ""
    )
    requested_end_date = parsed_anchor.isoformat() if parsed_anchor else str(anchor_date or "")
    chunks: List[SplitDetailSqlChunk] = []
    for index in range(scheduled_chunk_count):
        offset = index * window_days
        upper = min(total_days, offset + window_days)
        restriction = split_detail_chunk_restriction_sql(
            partition_column,
            anchor_expr,
            offset,
            upper - 1,
        )
        chunk_sql = add_sql_where_condition(text, restriction)
        chunk_sql = replace_sql_limit(chunk_sql, capped_limit)
        chunk_start = (
            (parsed_anchor - timedelta(days=upper - 1)).isoformat()
            if parsed_anchor
            else ""
        )
        chunk_end = (
            (parsed_anchor - timedelta(days=offset)).isoformat()
            if parsed_anchor
            else ""
        )
        chunks.append(
            SplitDetailSqlChunk(
                index=index + 1,
                offset_start_days=offset,
                offset_end_days=upper - 1,
                start_date=chunk_start,
                end_date=chunk_end,
                sql=chunk_sql,
            )
        )
    return SplitDetailSqlPlan(
        requested_days=total_days,
        requested_start_date=requested_start_date,
        requested_end_date=requested_end_date,
        anchor_date=requested_end_date,
        chunk_days=window_days,
        max_chunks=chunk_cap,
        required_chunk_count=required_chunk_count,
        chunks=tuple(chunks),
    )


def split_detail_sql_chunk_contract_error(
    base_sql: str,
    plan: SplitDetailSqlPlan,
    chunk: SplitDetailSqlChunk,
    time_column: str,
    limit: int,
) -> Tuple[str, str]:
    """Verify a split chunk is only a governed restriction of validated base SQL.

    This contract deliberately does not replace the ordinary time-window gate.
    Callers must first validate ``base_sql`` against the full requested window.
    A chunk is then accepted only when its plan metadata is internally complete
    and its SQL is exactly the base query plus two mandatory root-AND bounds on
    the declared partition column.
    """

    partition_column = normalize_identifier(time_column)
    if not str(base_sql or "").strip() or not partition_column or int(limit or 0) <= 0:
        return "SPLIT_WINDOW_PLAN_INVALID", "split validation requires base SQL, partitionColumn, and a positive limit"
    if plan.requested_days <= 0 or plan.chunk_days <= 0 or plan.max_chunks <= 0:
        return "SPLIT_WINDOW_PLAN_INVALID", "split window sizes and caps must be positive"
    expected_required_count = (plan.requested_days + plan.chunk_days - 1) // plan.chunk_days
    expected_planned_count = min(expected_required_count, plan.max_chunks)
    if plan.required_chunk_count != expected_required_count or len(plan.chunks) != expected_planned_count:
        return "SPLIT_WINDOW_PLAN_INVALID", "split plan does not preserve the full requested-window obligation"

    parsed_anchor: Optional[date] = None
    if plan.anchor_date:
        try:
            parsed_anchor = date.fromisoformat(plan.anchor_date)
        except ValueError:
            return "SPLIT_WINDOW_PLAN_INVALID", "split plan anchorDate is not an ISO date"
        expected_requested_start = (parsed_anchor - timedelta(days=plan.requested_days - 1)).isoformat()
        if (
            plan.requested_start_date != expected_requested_start
            or plan.requested_end_date != parsed_anchor.isoformat()
        ):
            return "SPLIT_WINDOW_PLAN_INVALID", "split plan requestedWindow does not match anchorDate and requestedDays"
    elif plan.requested_start_date or plan.requested_end_date:
        return "SPLIT_WINDOW_PLAN_INVALID", "split plan has requested dates without a valid anchorDate"

    for position, planned_chunk in enumerate(plan.chunks, start=1):
        expected_offset_start = (position - 1) * plan.chunk_days
        expected_offset_end = min(plan.requested_days, expected_offset_start + plan.chunk_days) - 1
        if (
            planned_chunk.index != position
            or planned_chunk.offset_start_days != expected_offset_start
            or planned_chunk.offset_end_days != expected_offset_end
        ):
            return "SPLIT_WINDOW_PLAN_INVALID", "split chunk offsets are not a contiguous partition of the requested window"
        if parsed_anchor:
            expected_start_date = (parsed_anchor - timedelta(days=expected_offset_end)).isoformat()
            expected_end_date = (parsed_anchor - timedelta(days=expected_offset_start)).isoformat()
            if planned_chunk.start_date != expected_start_date or planned_chunk.end_date != expected_end_date:
                return "SPLIT_WINDOW_PLAN_INVALID", "split chunk date metadata does not match its governed offsets"
        elif planned_chunk.start_date or planned_chunk.end_date:
            return "SPLIT_WINDOW_PLAN_INVALID", "split chunk dates require a valid plan anchorDate"

    if chunk.index <= 0 or chunk.index > len(plan.chunks) or plan.chunks[chunk.index - 1] != chunk:
        return "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID", "split SQL chunk is not the corresponding member of the validated plan"

    escaped_anchor = str(plan.anchor_date or "").replace("'", "''")
    anchor_expression = "'%s'" % escaped_anchor if escaped_anchor else "CURDATE()"
    restriction_sql = split_detail_chunk_restriction_sql(
        partition_column,
        anchor_expression,
        chunk.offset_start_days,
        chunk.offset_end_days,
    )
    expected_sql = replace_sql_limit(
        add_sql_where_condition(base_sql, restriction_sql),
        int(limit),
    )
    actual_ast = parse_sql_for_binding(chunk.sql)
    expected_ast = parse_sql_for_binding(expected_sql)
    base_ast = parse_sql_for_binding(replace_sql_limit(base_sql, int(limit)))
    restriction_ast = parse_sql_for_binding("SELECT * FROM `__split_scope` WHERE %s" % restriction_sql)
    if not all(isinstance(item, exp.Select) for item in (actual_ast, expected_ast, base_ast, restriction_ast)):
        return "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID", "split SQL cannot be parsed as a single SELECT"

    def canonical(expression: exp.Expression) -> str:
        return expression.sql(dialect="doris", identify=True, pretty=False)

    if canonical(actual_ast) != canonical(expected_ast):
        return (
            "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID",
            "split SQL must be exactly the validated base SQL plus its governed partition restriction",
        )

    def where_terms(select: exp.Select) -> List[exp.Expression]:
        where = select.args.get("where")
        return root_and_terms(where.this) if isinstance(where, exp.Where) else []

    base_terms = Counter(canonical(term) for term in where_terms(base_ast))
    restriction_terms = where_terms(restriction_ast)
    actual_terms = Counter(canonical(term) for term in where_terms(actual_ast))
    expected_terms = base_terms + Counter(canonical(term) for term in restriction_terms)
    if len(restriction_terms) != 2 or actual_terms != expected_terms:
        return (
            "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID",
            "split partition bounds must be mandatory-positive root-AND predicates",
        )
    normalized_bounds = [unwrap_boolean_parentheses(term) for term in restriction_terms]
    if (
        not any(
            isinstance(term, exp.GTE) and predicate_column_name(term.this) == partition_column
            for term in normalized_bounds
        )
        or not any(
            isinstance(term, exp.LT) and predicate_column_name(term.this) == partition_column
            for term in normalized_bounds
        )
        or any(direct_expression_columns(term) != [partition_column] for term in normalized_bounds)
    ):
        return (
            "SPLIT_WINDOW_CHUNK_DERIVATION_INVALID",
            "split SQL bounds must restrict only the governed partitionColumn",
        )
    return "", ""


def split_window_coverage_contract(
    plan: SplitDetailSqlPlan,
    chunk_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Materialize a generic, auditable coverage contract for split execution."""

    normalized_outcomes: List[Dict[str, Any]] = []
    seen_indexes = set()
    for raw in sorted(chunk_outcomes, key=lambda item: int(item.get("chunkIndex") or 0)):
        index = int(raw.get("chunkIndex") or 0)
        if index <= 0 or index in seen_indexes:
            continue
        seen_indexes.add(index)
        normalized_outcomes.append(
            {
                key: value
                for key, value in {
                    "chunkIndex": index,
                    "status": str(raw.get("status") or "failed"),
                    "rows": int(raw.get("rows") or 0),
                    "errorCode": str(raw.get("errorCode") or ""),
                    "error": str(raw.get("error") or "")[:240],
                }.items()
                if value not in ("", None)
            }
        )
    succeeded = [item for item in normalized_outcomes if item.get("status") == "succeeded"]
    failed = [item for item in normalized_outcomes if item.get("status") != "succeeded"]
    planned_indexes = {chunk.index for chunk in plan.chunks}
    executed_indexes = {int(item["chunkIndex"]) for item in normalized_outcomes}
    unexecuted_indexes = sorted(planned_indexes - executed_indexes)
    complete = bool(
        not plan.truncated
        and not failed
        and not unexecuted_indexes
        and len(succeeded) == plan.required_chunk_count
    )
    reason_parts: List[str] = []
    if plan.truncated:
        reason_parts.append(
            "maxChunks scheduled %d of %d required chunks"
            % (len(plan.chunks), plan.required_chunk_count)
        )
    if failed:
        reason_parts.append("%d executed chunks failed" % len(failed))
    if unexecuted_indexes:
        reason_parts.append("planned chunks not executed: %s" % ",".join(map(str, unexecuted_indexes)))
    return {
        "contractVersion": "split_window_coverage_v1",
        "code": "" if complete else "SPLIT_WINDOW_COVERAGE_INCOMPLETE",
        "requestedWindow": {
            "startDate": plan.requested_start_date,
            "endDate": plan.requested_end_date,
            "days": plan.requested_days,
            "anchorDate": plan.anchor_date,
        },
        "chunkDays": plan.chunk_days,
        "maxChunks": plan.max_chunks,
        "requiredChunkCount": plan.required_chunk_count,
        "plannedChunkCount": len(plan.chunks),
        "plannedChunks": [chunk.contract_payload() for chunk in plan.chunks],
        "executedChunkCount": len(normalized_outcomes),
        "executedChunks": normalized_outcomes,
        "succeededChunkCount": len(succeeded),
        "succeededChunks": succeeded,
        "failedChunkCount": len(failed),
        "failedChunks": failed,
        "unexecutedChunkIndexes": unexecuted_indexes,
        "omittedChunkCount": plan.omitted_chunk_count,
        "truncated": plan.truncated,
        "complete": complete,
        "lineageComplete": complete,
        "reason": "; ".join(reason_parts),
    }


def split_detail_sql_by_time_windows(
    sql: str,
    days: int,
    chunk_days: int,
    max_chunks: int,
    limit: int,
    time_column: str,
    anchor_date: str = "",
) -> List[str]:
    plan = build_split_detail_sql_plan(
        sql,
        days,
        chunk_days,
        max_chunks,
        limit,
        time_column,
        anchor_date=anchor_date,
    )
    return [chunk.sql for chunk in plan.chunks] if plan else []


def add_sql_where_condition(sql: str, condition: str) -> str:
    text = str(sql or "").strip()
    parsed = parse_sql_for_binding(text)
    try:
        restriction = sqlglot.parse_one(
            str(condition or "").strip(),
            read="doris",
            into=exp.Condition,
        )
    except Exception as exc:
        raise ValueError("SQL_AST_RESTRICTION_PARSE_FAILED") from exc
    if parsed is None or restriction is None:
        raise ValueError("SQL_AST_WHERE_BINDING_PARSE_FAILED")
    where = parsed.args.get("where")
    if isinstance(where, exp.Where):
        predicate = exp.and_(where.this.copy(), restriction.copy())
    else:
        predicate = restriction.copy()
    parsed.set("where", exp.Where(this=predicate))
    return normalize_ast_bound_sql_text(
        parsed.sql(dialect="mysql", identify=True).replace("?", "%s")
    )


def replace_sql_limit(sql: str, limit: int) -> str:
    text = str(sql or "").strip()
    value = max(1, int(limit or 1))
    parsed = parse_sql_for_binding(text)
    if parsed is None:
        raise ValueError("SQL_AST_LIMIT_BINDING_PARSE_FAILED")
    parsed.set("limit", exp.Limit(expression=exp.Literal.number(value)))
    return normalize_ast_bound_sql_text(
        parsed.sql(dialect="mysql", identify=True).replace("?", "%s")
    )


def quote_identifier(column: str) -> str:
    return "`%s`" % str(column).replace("`", "")


def sql_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'%s'" % str(value).replace("'", "''")


def normalize_identifier(value: Any) -> str:
    return str(value or "").strip().strip("`").lower()


def split_filter_values(value: Any, max_values: int = 200) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw = str(value or "").strip()
        if not raw:
            return []
        raw_values = [
            item.strip()
            for item in raw.replace("，", ",").split(",")
        ]
    values: List[Any] = []
    for item in raw_values:
        if blank_entity_value(item):
            continue
        if item not in values:
            values.append(item)
        if len(values) >= max_values:
            break
    return values


def parse_sql_literal(value: str) -> Any:
    text = str(value or "").strip()
    if not text:
        return ""
    if text in {"%s", "?"} or _is_named_placeholder(text):
        return ""
    if text.upper() == "NULL":
        return None
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if _is_ascii_integer(text):
        try:
            return int(text)
        except ValueError:
            return text
    if _is_ascii_decimal(text):
        try:
            return float(text)
        except ValueError:
            return text
    return text


def parse_sql_literal_values(rhs: str) -> List[Any]:
    text = str(rhs or "").strip()
    if not text:
        return []
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    values: List[Any] = []
    current = []
    in_quote = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "'":
            current.append(char)
            if in_quote and index + 1 < len(text) and text[index + 1] == "'":
                current.append(text[index + 1])
                index += 2
                continue
            in_quote = not in_quote
        elif char == "," and not in_quote:
            parsed = parse_sql_literal("".join(current).strip())
            if not blank_entity_value(parsed):
                values.append(parsed)
            current = []
        else:
            current.append(char)
        index += 1
    parsed = parse_sql_literal("".join(current).strip())
    if not blank_entity_value(parsed):
        values.append(parsed)
    return values


def add_bind_values(target: Dict[str, List[Any]], column: str, values: List[Any], max_values: int = 200) -> None:
    key = normalize_identifier(column)
    if not key:
        return
    bucket = target.setdefault(key, [])
    for value in values:
        if len(bucket) >= max_values:
            break
        if blank_entity_value(value) or value in bucket:
            continue
        bucket.append(value)


def mandatory_semantic_filter_nodes(intent: QuestionIntent) -> List[Any]:
    """Return predicates that every row must satisfy in the semantic filter tree.

    Only root predicates and descendants of AND groups are mandatory-positive.
    A predicate below OR/NOT cannot safely drive global parameter binding because
    replacing every predicate on the same column would change boolean semantics.
    """

    spec = getattr(intent, "semantic_query", None)
    nodes = list(getattr(spec, "filter_nodes", None) or [])
    root_id = str(getattr(spec, "root_filter_node_id", "") or "")
    node_by_id = {str(getattr(node, "node_id", "") or ""): node for node in nodes}
    if not root_id or root_id not in node_by_id:
        return []

    def visit(node_id: str, visiting: set[str]) -> List[Any]:
        if node_id in visiting:
            return []
        node = node_by_id.get(node_id)
        if node is None:
            return []
        node_type = str(getattr(node, "node_type", "") or "").lower()
        if node_type == "predicate":
            return [node]
        if node_type != "group" or str(getattr(node, "logical_operator", "") or "").lower() != "and":
            return []
        result: List[Any] = []
        next_visiting = {*visiting, node_id}
        for child_id in getattr(node, "child_node_ids", None) or []:
            result.extend(visit(str(child_id or ""), next_visiting))
        return result

    return visit(root_id, set())


def semantic_bind_values_by_column(intent: QuestionIntent, columns: set, max_values: int = 200) -> Dict[str, List[Any]]:
    """Bind only unambiguous mandatory EQ/IN predicates by physical column."""

    candidates: Dict[str, List[Any]] = {}
    counts: Dict[str, int] = {}
    for node in mandatory_semantic_filter_nodes(intent):
        if str(getattr(node, "member_kind", "") or "").lower() == "measure":
            continue
        if str(getattr(node, "resolution_status", "") or "").lower() != "resolved":
            continue
        if str(getattr(node, "operator", "") or "").lower() not in {"eq", "in"}:
            continue
        column = normalize_identifier(getattr(node, "bound_field", ""))
        if not column or column not in columns:
            continue
        counts[column] = counts.get(column, 0) + 1
        candidates.setdefault(column, []).extend(list(getattr(node, "resolved_values", None) or []))
    return {
        column: list(dict.fromkeys(values))[:max_values]
        for column, values in candidates.items()
        if counts.get(column) == 1 and values
    }


def node_bind_values_by_column(
    intent: QuestionIntent,
    asset_pack: PlanningAssetPack,
    columns: set,
    context: NodeExecutionContext,
    max_values: int = 200,
) -> Dict[str, List[Any]]:
    normalized_columns = {normalize_identifier(column) for column in columns}
    values_by_column: Dict[str, List[Any]] = {}
    merchant_id = str(context.merchant_id or "").strip()
    tenant_column = declared_tenant_column(intent, asset_pack, context, normalized_columns)
    if merchant_id and tenant_column:
        add_bind_values(values_by_column, tenant_column, [merchant_id], 1)
    region_column = normalize_identifier((context.context_package or {}).get("regionFilterColumn"))
    if context.authorized_region and region_column in normalized_columns:
        add_bind_values(values_by_column, region_column, [context.authorized_region], 1)
    store_column = normalize_identifier((context.context_package or {}).get("storeFilterColumn"))
    if context.authorized_store_ids and store_column in normalized_columns:
        add_bind_values(values_by_column, store_column, list(context.authorized_store_ids), max_values)
    for column, values in semantic_bind_values_by_column(intent, normalized_columns, max_values).items():
        add_bind_values(values_by_column, column, values, max_values)
    filter_column = normalize_identifier(intent.filter_column)
    if filter_column in normalized_columns:
        reference = getattr(intent, "entity_reference", None)
        governed_values = (
            list(getattr(reference, "values", None) or [])
            if str(getattr(reference, "status", "") or "") == "resolved"
            and normalize_identifier(getattr(reference, "field", "")) == filter_column
            else split_filter_values(intent.filter_value, max_values)
        )
        add_bind_values(values_by_column, filter_column, governed_values, max_values)
    for entity_set in context.upstream_entity_sets or []:
        join_key = normalize_identifier(entity_set.join_key)
        if join_key in normalized_columns:
            add_bind_values(values_by_column, join_key, list(entity_set.values or []), max_values)
        for column, values in (entity_set.column_values or {}).items():
            normalized = normalize_identifier(column)
            if normalized in normalized_columns:
                add_bind_values(values_by_column, normalized, list(values or []), max_values)
    return values_by_column


def declared_tenant_column(
    intent: QuestionIntent,
    asset_pack: PlanningAssetPack,
    context: NodeExecutionContext,
    columns: set,
) -> str:
    package = dict(context.context_package or {})
    table_contracts = package.get("tableContracts") if isinstance(package.get("tableContracts"), dict) else {}
    table_contract = table_contracts.get(intent.preferred_table) if isinstance(table_contracts, dict) else {}
    asset_metadata: Dict[str, Any] = {}
    for item in asset_pack.tables:
        if (item.table or item.key) == intent.preferred_table:
            asset_metadata = dict(item.metadata or {})
            break
    row_policy = asset_metadata.get("rowAccessPolicy") if isinstance(asset_metadata.get("rowAccessPolicy"), dict) else {}
    candidates = [
        asset_metadata.get("merchantFilterColumn"),
        asset_metadata.get("tenantFilterColumn"),
        row_policy.get("filterColumn") if isinstance(row_policy, dict) else "",
        package.get("merchantFilterColumn"),
        package.get("tenantFilterColumn"),
        (table_contract or {}).get("merchantFilterColumn") if isinstance(table_contract, dict) else "",
        (table_contract or {}).get("tenantFilterColumn") if isinstance(table_contract, dict) else "",
    ]
    return next((normalize_identifier(item) for item in candidates if normalize_identifier(item) in columns), "")


def append_note(existing: str, note: str) -> str:
    parts = [str(existing or "").strip(), str(note or "").strip()]
    return "; ".join(part for part in parts if part)


def realtime_fallback_for_table(asset_pack: PlanningAssetPack, table: str) -> Optional[Any]:
    normalized_table = normalize_identifier(table)
    known_tables = set(asset_pack.known_tables())
    for item in asset_pack.realtime_fallbacks or []:
        metadata = item.metadata or {}
        fallback_table = str(
            item.table
            or metadata.get("realtimeTable")
            or metadata.get("fallbackTable")
            or metadata.get("targetTable")
            or metadata.get("table")
            or ""
        )
        source_candidates = {
            normalize_identifier(str(item.key or "")),
            normalize_identifier(str(metadata.get("sourceTable") or "")),
            normalize_identifier(str(metadata.get("offlineTable") or "")),
            normalize_identifier(str(metadata.get("baseTable") or "")),
            normalize_identifier(str(metadata.get("ownerTable") or "")),
        }
        fallback_candidates = {
            normalize_identifier(fallback_table),
            normalize_identifier(str(item.table or "")),
        }
        if normalized_table in source_candidates and fallback_table in known_tables:
            return item
        description = ("%s %s" % (item.title, item.description)).lower()
        if (
            not any(source_candidates)
            and normalized_table
            and normalized_table not in fallback_candidates
            and fallback_table in known_tables
            and normalized_table in description
        ):
            return item
    return None


def partition_is_stale_for_near_realtime(max_pt: str, requested_days: int) -> bool:
    if int(requested_days or 0) > 2:
        return False
    parsed = parse_partition_date(max_pt)
    if not parsed:
        return False
    return parsed < (date.today() - timedelta(days=1))


def parse_partition_date(value: str) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    for pattern in ("%Y%m%d", "%Y-%m-%d"):
        candidate = text[:8] if pattern == "%Y%m%d" else text[:10]
        try:
            return datetime.strptime(candidate, pattern).date()
        except Exception:
            continue
    return None


def predicate_column_name(expression: Any) -> str:
    if isinstance(expression, exp.Column):
        return normalize_identifier(expression.name)
    if isinstance(expression, exp.Identifier):
        return normalize_identifier(expression.name)
    return ""


def parse_sql_for_binding(sql: str) -> Optional[exp.Expression]:
    try:
        return sqlglot.parse_one((sql or "").strip(), read="doris")
    except Exception:
        try:
            return sqlglot.parse_one((sql or "").replace("%s", "?").strip(), read="mysql")
        except Exception:
            return None


BOUND_PARAMETER_PREFIX = "__contract_param_"


def parse_bound_sql_for_contract(sql: str) -> Optional[exp.Expression]:
    """Parse DB-API ``%s`` SQL while preserving each parameter position in AST."""

    index = 0
    source = str(sql or "")
    output: List[str] = []
    position = 0
    quote = ""
    line_comment = False
    block_comment = False
    while position < len(source):
        if line_comment:
            output.append(source[position])
            if source[position] in "\r\n":
                line_comment = False
            position += 1
            continue
        if block_comment:
            if source.startswith("*/", position):
                output.append("*/")
                block_comment = False
                position += 2
            else:
                output.append(source[position])
                position += 1
            continue
        if quote:
            character = source[position]
            output.append(character)
            if character == "\\" and position + 1 < len(source):
                output.append(source[position + 1])
                position += 2
                continue
            if character == quote:
                if position + 1 < len(source) and source[position + 1] == quote:
                    output.append(source[position + 1])
                    position += 2
                    continue
                quote = ""
            position += 1
            continue
        if source.startswith("--", position):
            output.append("--")
            line_comment = True
            position += 2
            continue
        if source.startswith("/*", position):
            output.append("/*")
            block_comment = True
            position += 2
            continue
        if source[position] in {"'", '"', "`"}:
            quote = source[position]
            output.append(source[position])
            position += 1
            continue
        if source.startswith("%s", position):
            output.append(":%s%d" % (BOUND_PARAMETER_PREFIX, index))
            index += 1
            position += 2
            continue
        output.append(source[position])
        position += 1
    normalized = "".join(output)
    try:
        return sqlglot.parse_one(normalized.strip(), read="doris")
    except Exception:
        try:
            return sqlglot.parse_one(normalized.strip(), read="mysql")
        except Exception:
            return None


def bound_parameter_index(expression: Any) -> Optional[int]:
    if not isinstance(expression, exp.Placeholder):
        return None
    name = str(expression.this or "")
    if not name.startswith(BOUND_PARAMETER_PREFIX):
        return None
    suffix = name[len(BOUND_PARAMETER_PREFIX) :]
    return int(suffix) if suffix.isdigit() else None


def unwrap_boolean_parentheses(expression: exp.Expression) -> exp.Expression:
    current = expression
    while isinstance(current, exp.Paren):
        current = current.this
    return current


def root_and_terms(expression: exp.Expression) -> List[exp.Expression]:
    current = unwrap_boolean_parentheses(expression)
    if isinstance(current, exp.And):
        return root_and_terms(current.this) + root_and_terms(current.expression)
    return [current]


def direct_expression_columns(expression: exp.Expression) -> List[str]:
    """Return columns in an expression without descending into nested SELECTs."""

    columns: List[str] = []

    def visit(node: Any, root: bool = False) -> None:
        if not isinstance(node, exp.Expression):
            return
        if not root and isinstance(node, (exp.Select, exp.Subquery)):
            return
        if isinstance(node, exp.Column):
            columns.append(normalize_identifier(node.name))
            return
        for child in node.iter_expressions():
            visit(child)

    visit(expression, root=True)
    return columns


def select_scopes_reading_table(parsed: exp.Expression, table: str = "") -> List[Any]:
    target = normalize_identifier(table)
    scopes = list(traverse_scope(parsed))
    if target:
        return [
            scope
            for scope in scopes
            if any(
                isinstance(pair[1], exp.Table) and normalize_identifier(pair[1].name) == target
                for pair in (getattr(scope, "selected_sources", {}) or {}).values()
            )
        ]
    if isinstance(parsed, exp.Select):
        return [scope for scope in scopes if scope.expression is parsed]
    outer = parsed.find(exp.Select)
    return [scope for scope in scopes if scope.expression is outer] if outer is not None else []


def mandatory_column_predicate(
    select: exp.Expression,
    column: str,
    *,
    require_placeholders: bool = False,
) -> Tuple[Optional[exp.Expression], List[int]]:
    """Find the single positive EQ/IN predicate on the root-AND path."""

    target = normalize_identifier(column)
    where = select.args.get("where") if isinstance(select, exp.Select) else None
    condition = where.this if isinstance(where, exp.Where) else None
    if not target or condition is None:
        return None, []
    references = [term for term in root_and_terms(condition) if target in direct_expression_columns(term)]
    if len(references) != 1:
        return None, []
    predicate = unwrap_boolean_parentheses(references[0])
    if not isinstance(predicate, (exp.EQ, exp.In)):
        return None, []
    if predicate_column_name(predicate.this) != target:
        return None, []
    if isinstance(predicate, exp.In):
        if predicate.args.get("query") is not None or predicate.find(exp.Subquery) is not None:
            return None, []
        rhs = list(predicate.expressions or [])
        if not rhs:
            return None, []
    else:
        if predicate.expression.find(exp.Select) is not None or predicate.expression.find(exp.Subquery) is not None:
            return None, []
        rhs = [predicate.expression]
    indexes = [bound_parameter_index(item) for item in rhs]
    if require_placeholders and any(index is None for index in indexes):
        return None, []
    return predicate, [int(index) for index in indexes if index is not None]


def sql_has_mandatory_column_filter(sql: str, column: str, preferred_table: str = "") -> bool:
    parsed = parse_bound_sql_for_contract(sql)
    if parsed is None:
        return False
    scopes = select_scopes_reading_table(parsed, preferred_table)
    return bool(scopes) and all(
        mandatory_column_predicate(scope.expression, column)[0] is not None
        for scope in scopes
    )


def sql_has_bound_scope_column_values(
    sql: str,
    column: str,
    params: List[Any],
    expected_values: List[Any],
    preferred_table: str = "",
) -> bool:
    """Verify exact bound values in every base-table SELECT scope."""

    parsed = parse_bound_sql_for_contract(sql)
    if parsed is None:
        return False
    scopes = select_scopes_reading_table(parsed, preferred_table)
    if not scopes:
        return False
    expected = sorted(str(item) for item in expected_values)
    for scope in scopes:
        predicate, indexes = mandatory_column_predicate(
            scope.expression,
            column,
            require_placeholders=True,
        )
        if predicate is None or not indexes or any(index >= len(params) for index in indexes):
            return False
        actual = sorted(str(params[index]) for index in indexes)
        if actual != expected:
            return False
    return True


def placeholder_expression() -> exp.Placeholder:
    return exp.Placeholder()


def sql_expression_is_placeholder(expression: Any) -> bool:
    return isinstance(expression, exp.Placeholder) or str(expression or "").strip() in {"?", "%s"}


def has_merchant_filter_predicate(sql: str, columns: set) -> bool:
    merchant_columns = sorted({normalize_identifier(item) for item in columns if normalize_identifier(item)})
    parsed = parse_sql_for_binding(sql)
    if parsed is not None:
        for predicate in list(parsed.find_all(exp.EQ)) + list(parsed.find_all(exp.In)):
            if predicate_column_name(predicate.this) in merchant_columns:
                return True
    return False


def sql_has_bound_merchant_filter(sql: str, columns: set) -> bool:
    merchant_columns = sorted({normalize_identifier(item) for item in columns if normalize_identifier(item)})
    if not merchant_columns:
        return False
    parsed = parse_sql_for_binding(sql)
    if parsed is not None:
        for predicate in parsed.find_all(exp.EQ):
            if predicate_column_name(predicate.this) in merchant_columns and sql_expression_is_placeholder(predicate.expression):
                return True
        for predicate in parsed.find_all(exp.In):
            if predicate_column_name(predicate.this) in merchant_columns and any(sql_expression_is_placeholder(item) for item in predicate.expressions):
                return True
    return False


def sql_has_bound_column_filter(sql: str, column: str) -> bool:
    normalized = normalize_identifier(column)
    if not normalized:
        return False
    parsed = parse_sql_for_binding(sql)
    if parsed is not None:
        for predicate in list(parsed.find_all(exp.EQ)) + list(parsed.find_all(exp.In)):
            if predicate_column_name(predicate.this) != normalized:
                continue
            expressions = [predicate.expression] if isinstance(predicate, exp.EQ) else list(predicate.expressions or [])
            if any(sql_expression_is_placeholder(item) for item in expressions):
                return True
    return False


def predicate_references_column(expression: Any, column: str) -> bool:
    normalized = normalize_identifier(column)
    if not normalized:
        return False
    for item in expression.find_all(exp.Column):
        if normalize_identifier(item.name) == normalized:
            return True
    return False


def sql_references_column(sql: str, column: str) -> bool:
    normalized = normalize_identifier(column)
    if not normalized:
        return False
    parsed = parse_sql_for_binding(sql)
    if parsed is not None:
        return any(normalize_identifier(item.name) == normalized for item in parsed.find_all(exp.Column))
    return False


def bind_node_sql_parameters_ast(
    sql: str,
    values_by_column: Dict[str, List[Any]],
    merchant_columns: set,
    context: NodeExecutionContext,
) -> Tuple[str, List[Any], str, bool]:
    parsed = parse_sql_for_binding(sql)
    if parsed is None:
        return sql, [], "", False
    params: List[Any] = []
    merchant_bound = not merchant_columns
    bound_any = False
    binding_error = ""

    def values_for(column: str, original_values: List[Any]) -> List[Any]:
        values = list(values_by_column.get(column) or original_values or [])
        values = [value for value in values if not blank_entity_value(value)]
        if column in merchant_columns:
            return [context.merchant_id] if not blank_entity_value(context.merchant_id) else []
        return values

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal merchant_bound, bound_any, binding_error
        if isinstance(node, exp.EQ):
            column = predicate_column_name(node.this)
            if column not in values_by_column and column not in merchant_columns:
                return node
            original_values = parse_ast_literal_values([node.expression])
            values = values_for(column, original_values)
            if not values:
                return node
            if len(values) > 1:
                placeholders = [placeholder_expression() for _ in values]
                params.extend(values)
                bound_any = True
                if column in merchant_columns:
                    merchant_bound = True
                return exp.In(this=node.this.copy(), expressions=placeholders)
            node.set("expression", placeholder_expression())
            params.append(values[0])
            bound_any = True
            if column in merchant_columns:
                merchant_bound = True
            return node
        if isinstance(node, exp.In):
            column = predicate_column_name(node.this)
            if column not in values_by_column and column not in merchant_columns:
                return node
            if node.args.get("query") is not None or node.find(exp.Subquery) is not None:
                binding_error = "受控过滤字段不能使用 IN 子查询，必须使用可绑定的标量值列表"
                return node
            original_values = parse_ast_literal_values(list(node.expressions or []))
            values = values_for(column, original_values)
            if not values:
                return node
            placeholders = [placeholder_expression() for _ in values]
            node.set("expressions", placeholders)
            params.extend(values)
            bound_any = True
            if column in merchant_columns:
                merchant_bound = True
            return node
        return node

    bound = parsed.transform(transform, copy=True)
    bound_sql = normalize_ast_bound_sql_text(bound.sql(dialect="mysql", identify=True).replace("?", "%s"))
    if binding_error:
        return bound_sql, params, binding_error, True
    placeholder_count = sum(1 for _ in bound.find_all(exp.Placeholder))
    if placeholder_count != len(params):
        return bound_sql, params, "SQL 占位符数量与受控绑定参数不一致", True
    if merchant_columns and not merchant_bound:
        return bound_sql, params, "SQL 缺少语义资产声明的租户过滤字段", True
    if not bound_any:
        return sql, [], "", True
    return bound_sql, params, "", True


def parse_ast_literal_values(expressions: List[Any]) -> List[Any]:
    values: List[Any] = []
    for expression in expressions:
        if isinstance(expression, exp.Literal):
            values.append(expression.this)
        elif isinstance(expression, exp.Tuple):
            values.extend(parse_ast_literal_values(list(expression.expressions or [])))
        elif sql_expression_is_placeholder(expression):
            continue
        else:
            parsed = parse_sql_literal(str(expression))
            if not blank_entity_value(parsed):
                values.append(parsed)
    return values


def normalize_ast_bound_sql_text(sql: str) -> str:
    text = str(sql or "")
    text = _apply_text_replacements(
        text,
        _date_sub_replacements(text),
    )
    return _apply_text_replacements(
        text,
        _null_and_empty_string_replacements(text),
    )


def bind_node_sql_parameters(
    sql: str,
    intent: QuestionIntent,
    asset_pack: PlanningAssetPack,
    context: NodeExecutionContext,
    max_filter_values: int = 0,
) -> Tuple[str, List[Any], str]:
    reference = getattr(intent, "entity_reference", None)
    if (
        str(getattr(reference, "status", "") or "") == "resolved"
        and normalize_identifier(getattr(reference, "field", "")) == normalize_identifier(intent.filter_column)
    ):
        governed_values = list(getattr(reference, "values", None) or [])
        if max_filter_values > 0 and len(governed_values) > max_filter_values:
            return (
                sql,
                [],
                "ENTITY_FILTER_VALUE_LIMIT_EXCEEDED: entity filter contains %s values; limit=%s"
                % (len(governed_values), max_filter_values),
            )
    columns = {normalize_identifier(column) for column in asset_pack.known_columns(intent.preferred_table)}
    values_by_column = node_bind_values_by_column(intent, asset_pack, columns, context)
    merchant_columns = {declared_tenant_column(intent, asset_pack, context, columns)} - {""}
    if not values_by_column and not merchant_columns:
        return sql, [], ""
    ast_sql, ast_params, ast_error, ast_used = bind_node_sql_parameters_ast(sql, values_by_column, merchant_columns, context)
    if ast_used:
        return ast_sql, ast_params, ast_error
    return sql, [], "SQL_AST_BINDING_PARSE_FAILED"


def _is_named_placeholder(value: str) -> bool:
    if not value.startswith(":") or len(value) < 2:
        return False
    name = value[1:]
    if not (
        "A" <= name[0] <= "Z"
        or "a" <= name[0] <= "z"
        or name[0] == "_"
    ):
        return False
    return all(
        "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or character == "_"
        for character in name[1:]
    )


def _is_ascii_integer(value: str) -> bool:
    digits = value[1:] if value.startswith("-") else value
    return bool(digits) and all("0" <= character <= "9" for character in digits)


def _is_ascii_decimal(value: str) -> bool:
    unsigned = value[1:] if value.startswith("-") else value
    whole, separator, fraction = unsigned.partition(".")
    return bool(
        separator
        and whole
        and fraction
        and all("0" <= character <= "9" for character in whole)
        and all("0" <= character <= "9" for character in fraction)
    )


def _date_sub_replacements(
    text: str,
    *,
    expected_days: int | None = None,
    replacement_days: int | None = None,
) -> List[Tuple[int, int, str]]:
    try:
        tokens = Tokenizer().tokenize(text)
    except Exception:
        return []
    replacements: List[Tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.text.upper() != "DATE_SUB":
            index += 1
            continue
        cursor = index + 1
        if not _token_text_is(tokens, cursor, "("):
            index += 1
            continue
        cursor += 1
        current_date = _token_upper(tokens, cursor)
        if current_date not in {"CURDATE", "CURRENT_DATE"}:
            index += 1
            continue
        cursor += 1
        if _token_text_is(tokens, cursor, "("):
            if not _token_text_is(tokens, cursor + 1, ")"):
                index += 1
                continue
            cursor += 2
        if not _token_text_is(tokens, cursor, ","):
            index += 1
            continue
        cursor += 1
        if _token_upper(tokens, cursor) != "INTERVAL":
            index += 1
            continue
        cursor += 1
        if cursor >= len(tokens):
            break
        raw_days = str(tokens[cursor].text or "").strip()
        if not _is_ascii_integer(raw_days):
            index += 1
            continue
        observed_days = int(raw_days)
        if expected_days is not None and observed_days != expected_days:
            index += 1
            continue
        cursor += 1
        if _token_upper(tokens, cursor) != "DAY":
            index += 1
            continue
        cursor += 1
        if not _token_text_is(tokens, cursor, ")"):
            index += 1
            continue
        rendered_days = (
            observed_days
            if replacement_days is None
            else int(replacement_days)
        )
        replacements.append(
            (
                token.start,
                tokens[cursor].end + 1,
                "DATE_SUB(CURDATE(), INTERVAL %d DAY)" % rendered_days,
            )
        )
        index = cursor + 1
    return replacements


def _null_and_empty_string_replacements(
    text: str,
) -> List[Tuple[int, int, str]]:
    try:
        tokens = Tokenizer().tokenize(text)
    except Exception:
        return []
    replacements: List[Tuple[int, int, str]] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        upper = token.text.upper()
        if upper == "NOT":
            is_index = _qualified_identifier_end(tokens, index + 1)
            if (
                is_index > index + 1
                and _token_upper(tokens, is_index) == "IS"
                and _token_upper(tokens, is_index + 1) == "NULL"
            ):
                identifier = text[
                    tokens[index + 1].start : tokens[is_index].start
                ].rstrip()
                replacements.append(
                    (
                        token.start,
                        tokens[is_index + 1].end + 1,
                        "%s IS NOT NULL" % identifier,
                    )
                )
                index = is_index + 2
                continue
        if (
            upper == "<>"
            and index + 1 < len(tokens)
            and tokens[index + 1].text == ""
        ):
            replacements.append((token.start, token.end + 1, "!="))
        index += 1
    return replacements


def _qualified_identifier_end(tokens: List[Any], start: int) -> int:
    cursor = start
    observed_name = False
    while cursor < len(tokens):
        text = tokens[cursor].text
        upper = text.upper()
        if upper == "IS":
            return cursor if observed_name else start
        if text in {"`", "."}:
            cursor += 1
            continue
        if _is_identifier_token(text):
            observed_name = True
            cursor += 1
            continue
        return start
    return start


def _is_identifier_token(value: str) -> bool:
    if not value:
        return False
    first = value[0]
    if not (
        "A" <= first <= "Z"
        or "a" <= first <= "z"
        or first == "_"
    ):
        return False
    return all(
        "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or character == "_"
        for character in value[1:]
    )


def _token_text_is(tokens: List[Any], index: int, expected: str) -> bool:
    return bool(
        0 <= index < len(tokens)
        and str(tokens[index].text or "") == expected
    )


def _token_upper(tokens: List[Any], index: int) -> str:
    if index < 0 or index >= len(tokens):
        return ""
    return str(tokens[index].text or "").upper()


def _apply_text_replacements(
    text: str,
    replacements: List[Tuple[int, int, str]],
) -> str:
    result = text
    last_start = len(text) + 1
    for start, end, replacement in sorted(
        replacements,
        key=lambda item: item[0],
        reverse=True,
    ):
        if start < 0 or end < start or end > len(text) or end > last_start:
            continue
        result = result[:start] + replacement + result[end:]
        last_start = start
    return result


def blank_entity_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False
