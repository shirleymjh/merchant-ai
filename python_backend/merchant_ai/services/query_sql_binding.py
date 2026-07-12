from __future__ import annotations

import re
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


def split_detail_sql_by_pt_windows(sql: str, days: int, chunk_days: int, max_chunks: int, limit: int) -> List[str]:
    text = str(sql or "").strip()
    if not text or re.search(r"\b(group\s+by|union|join)\b", text, flags=re.I):
        return []
    if not re.search(r"\bpt\b|`pt`", text, flags=re.I):
        return []
    total_days = max(1, int(days or 0))
    window_days = max(1, int(chunk_days or 1))
    chunks = max(1, int(max_chunks or 1))
    capped_limit = max(1, int(limit or 1))
    result: List[str] = []
    for offset in range(0, total_days, window_days):
        if len(result) >= chunks:
            break
        upper = min(total_days, offset + window_days)
        lower = offset
        lower_bound = "`pt` >= DATE_SUB(CURDATE(), INTERVAL %d DAY)" % inclusive_day_interval(upper)
        if lower <= 0:
            upper_bound = "`pt` < DATE_ADD(CURDATE(), INTERVAL 1 DAY)"
        else:
            upper_bound = "`pt` < DATE_SUB(CURDATE(), INTERVAL %d DAY)" % inclusive_day_interval(lower)
        chunk_sql = add_sql_where_condition(text, "(%s AND %s)" % (lower_bound, upper_bound))
        chunk_sql = replace_sql_limit(chunk_sql, capped_limit)
        result.append(chunk_sql)
    return result


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


def is_dependent_context_column(column: str) -> bool:
    text = (column or "").lower()
    return any(
        token in text
        for token in [
            "status",
            "create_time",
            "close_time",
            "priority",
            "assignee",
            "operator",
            "type_code",
            "type_name",
        ]
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
        raw_values = [item.strip() for item in raw.split(",")] if "," in raw else [raw]
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


def node_bind_values_by_column(intent: QuestionIntent, columns: set, context: NodeExecutionContext, max_values: int = 200) -> Dict[str, List[Any]]:
    normalized_columns = {normalize_identifier(column) for column in columns}
    values_by_column: Dict[str, List[Any]] = {}
    merchant_id = str(context.merchant_id or "").strip()
    if merchant_id:
        for column in ["seller_id", "merchant_id"]:
            if column in normalized_columns:
                add_bind_values(values_by_column, column, [merchant_id], 1)
    region_column = normalize_identifier((context.context_package or {}).get("regionFilterColumn"))
    if context.authorized_region and region_column in normalized_columns:
        add_bind_values(values_by_column, region_column, [context.authorized_region], 1)
    store_column = normalize_identifier((context.context_package or {}).get("storeFilterColumn"))
    if context.authorized_store_ids and store_column in normalized_columns:
        add_bind_values(values_by_column, store_column, list(context.authorized_store_ids), max_values)
    filter_column = normalize_identifier(intent.filter_column)
    if filter_column in normalized_columns:
        add_bind_values(values_by_column, filter_column, split_filter_values(intent.filter_value, max_values), max_values)
    for entity_set in context.upstream_entity_sets or []:
        join_key = normalize_identifier(entity_set.join_key)
        if join_key in normalized_columns:
            add_bind_values(values_by_column, join_key, list(entity_set.values or []), max_values)
        for column, values in (entity_set.column_values or {}).items():
            normalized = normalize_identifier(column)
            if normalized in normalized_columns:
                add_bind_values(values_by_column, normalized, list(values or []), max_values)
    return values_by_column


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
    merchant_columns = [column for column in ["seller_id", "merchant_id", "shop_id"] if column in {normalize_identifier(item) for item in columns}]
    parsed = parse_sql_for_binding(sql)
    if parsed is not None:
        for predicate in list(parsed.find_all(exp.EQ)) + list(parsed.find_all(exp.In)):
            if predicate_column_name(predicate.this) in merchant_columns:
                return True
    pattern = bindable_predicate_pattern(merchant_columns)
    return bool(pattern.search(sql or "")) if pattern else False


def sql_has_bound_merchant_filter(sql: str, columns: set) -> bool:
    merchant_columns = [column for column in ["seller_id", "merchant_id", "shop_id"] if column in {normalize_identifier(item) for item in columns}]
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

    def values_for(column: str, original_values: List[Any]) -> List[Any]:
        values = list(values_by_column.get(column) or original_values or [])
        values = [value for value in values if not blank_entity_value(value)]
        if column in merchant_columns:
            return [context.merchant_id] if not blank_entity_value(context.merchant_id) else []
        return values

    def transform(node: exp.Expression) -> exp.Expression:
        nonlocal merchant_bound, bound_any
        if isinstance(node, exp.EQ):
            column = predicate_column_name(node.this)
            if column not in values_by_column and column not in merchant_columns:
                return node
            original_values = parse_ast_literal_values([node.expression])
            values = values_for(column, original_values)
            if not values:
                return node
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
    if merchant_columns and not merchant_bound:
        return bound_sql, params, "SQL 缺少可由后端绑定的商家过滤字段 seller_id/merchant_id/shop_id", True
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
) -> Tuple[str, List[Any], str]:
    columns = {normalize_identifier(column) for column in asset_pack.known_columns(intent.preferred_table)}
    values_by_column = node_bind_values_by_column(intent, columns, context)
    pattern = bindable_predicate_pattern(list(values_by_column.keys()))
    if not pattern:
        return sql, [], ""
    params: List[Any] = []
    merchant_columns = {"seller_id", "merchant_id", "shop_id"} & columns
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
        if op == "IN" and len(values) > 1:
            params.extend(values)
            return "%sIN (%s)" % (match.group("prefix"), ", ".join(["%s"] * len(values)))
        params.append(values[0])
        return "%s= %%s" % match.group("prefix")

    bound_sql = pattern.sub(replace, sql or "")
    if merchant_columns and not merchant_bound:
        return bound_sql, params, "SQL 缺少可由后端绑定的商家过滤字段 seller_id/merchant_id"
    return bound_sql, params, ""


def blank_entity_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False
