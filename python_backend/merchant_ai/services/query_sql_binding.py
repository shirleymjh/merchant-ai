from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

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
    pattern = re.compile(
        r"DATE_SUB\(\s*(CURDATE\(\)|CURRENT_DATE(?:\(\))?)\s*,\s*INTERVAL\s+'?%d'?\s+DAY\s*\)"
        % requested_days,
        flags=re.I,
    )
    return pattern.sub(lambda match: "DATE_SUB(%s, INTERVAL %d DAY)" % (match.group(1), inclusive_interval), text)


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
    if not text or re.search(r"\b(group\s+by|union|join)\b", text, flags=re.I):
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
    quoted_column = quote_identifier(partition_column)
    for index in range(scheduled_chunk_count):
        offset = index * window_days
        upper = min(total_days, offset + window_days)
        lower_bound = "%s >= DATE_SUB(%s, INTERVAL %d DAY)" % (
            quoted_column,
            anchor_expr,
            inclusive_day_interval(upper),
        )
        if offset <= 0:
            upper_bound = "%s < DATE_ADD(%s, INTERVAL 1 DAY)" % (quoted_column, anchor_expr)
        else:
            upper_bound = "%s < DATE_SUB(%s, INTERVAL %d DAY)" % (
                quoted_column,
                anchor_expr,
                inclusive_day_interval(offset),
            )
        chunk_sql = add_sql_where_condition(text, "(%s AND %s)" % (lower_bound, upper_bound))
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
    match = re.search(r"\s+(group\s+by|having|order\s+by|limit)\b", text, flags=re.I)
    tail = text[match.start() :] if match else ""
    head = text[: match.start()] if match else text
    if re.search(r"\swhere\s", head, flags=re.I):
        updated = "%s AND %s" % (head.rstrip(), condition)
    else:
        updated = "%s WHERE %s" % (head.rstrip(), condition)
    return (updated + tail).strip()


def replace_sql_limit(sql: str, limit: int) -> str:
    text = str(sql or "").strip()
    value = max(1, int(limit or 1))
    if re.search(r"\s+limit\s+\d+\s*$", text, flags=re.I):
        return re.sub(r"\s+limit\s+\d+\s*$", " LIMIT %d" % value, text, flags=re.I)
    return "%s LIMIT %d" % (text, value)


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
        raw_values = [item.strip() for item in re.split(r"[,，]", raw)] if re.search(r"[,，]", raw) else [raw]
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
    if text in {"%s", "?"} or re.match(r"^:[A-Za-z_][A-Za-z0-9_]*$", text):
        return ""
    if text.upper() == "NULL":
        return None
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        return text[1:-1].replace("''", "'")
    if re.match(r"^-?\d+$", text):
        try:
            return int(text)
        except ValueError:
            return text
    if re.match(r"^-?\d+\.\d+$", text):
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
        if blank_entity_value(value) or value in bucket:
            continue
        bucket.append(value)
        if len(bucket) >= max_values:
            break


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


def bindable_predicate_pattern(columns: List[str]) -> Optional[re.Pattern[str]]:
    names = [normalize_identifier(column) for column in columns if normalize_identifier(column)]
    if not names:
        return None
    column_pattern = "|".join(re.escape(column) for column in sorted(set(names), key=len, reverse=True))
    rhs_pattern = r"\([^)]*\)|%s|\?|:[A-Za-z_][A-Za-z0-9_]*|'(?:''|[^'])*'|-?\d+(?:\.\d+)?"
    return re.compile(
        r"(?P<prefix>`?(?P<column>%s)`?\s*)(?P<op>=|IN)\s*(?P<rhs>%s)" % (column_pattern, rhs_pattern),
        re.IGNORECASE,
    )


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
    pattern = bindable_predicate_pattern(merchant_columns)
    return bool(pattern.search(sql or "")) if pattern else False


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
    pattern = bindable_predicate_pattern(merchant_columns)
    if not pattern:
        return False
    for match in pattern.finditer(sql or ""):
        rhs = str(match.group("rhs") or "").strip()
        if "%s" in rhs or rhs == "?" or re.match(r"^:[A-Za-z_][A-Za-z0-9_]*$", rhs):
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
    pattern = bindable_predicate_pattern([normalized])
    if not pattern:
        return False
    return any("%s" in str(match.group("rhs") or "") or "?" in str(match.group("rhs") or "") for match in pattern.finditer(sql or ""))


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
    return bool(re.search(r"(?:`%s`|\b%s\b)" % (re.escape(normalized), re.escape(normalized)), sql or "", flags=re.I))


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
    text = re.sub(
        r"DATE_SUB\(CURRENT_DATE, INTERVAL '(\d+)' DAY\)",
        r"DATE_SUB(CURDATE(), INTERVAL \1 DAY)",
        sql or "",
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"NOT\s+((?:`[^`]+`\.)?`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*)\s+IS\s+NULL",
        r"\1 IS NOT NULL",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"((?:`[^`]+`\.)?`[^`]+`|[A-Za-z_][A-Za-z0-9_.]*)\s+<>\s+''",
        r"\1 != ''",
        text,
    )
    return text


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
    pattern = bindable_predicate_pattern(list(values_by_column.keys()))
    if not pattern:
        return sql, [], ""
    params: List[Any] = []
    merchant_columns = {declared_tenant_column(intent, asset_pack, context, columns)} - {""}
    ast_sql, ast_params, ast_error, ast_used = bind_node_sql_parameters_ast(sql, values_by_column, merchant_columns, context)
    if ast_used:
        return ast_sql, ast_params, ast_error
    merchant_bound = not merchant_columns

    def replace(match: re.Match[str]) -> str:
        nonlocal merchant_bound
        column = normalize_identifier(match.group("column"))
        op = str(match.group("op") or "=").upper()
        rhs = str(match.group("rhs") or "")
        values = list(values_by_column.get(column) or [])
        if not values:
            values = parse_sql_literal_values(rhs)
        values = [value for value in values if not blank_entity_value(value)]
        if not values:
            return match.group(0)
        if column in merchant_columns:
            values = [context.merchant_id]
            merchant_bound = True
        if len(values) > 1:
            params.extend(values)
            return "%sIN (%s)" % (match.group("prefix"), ", ".join(["%s"] * len(values)))
        params.append(values[0])
        return "%s= %%s" % match.group("prefix")

    bound_sql = pattern.sub(replace, sql or "")
    if merchant_columns and not merchant_bound:
        return bound_sql, params, "SQL 缺少语义资产声明的租户过滤字段"
    return bound_sql, params, ""


def blank_entity_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False
