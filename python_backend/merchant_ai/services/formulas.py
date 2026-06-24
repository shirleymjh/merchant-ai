from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set

import sqlglot
from sqlglot import exp


FORMULA_ALLOWED_TOKENS = {
    "SUM",
    "COUNT",
    "AVG",
    "MIN",
    "MAX",
    "DISTINCT",
    "CASE",
    "WHEN",
    "THEN",
    "ELSE",
    "END",
    "NULLIF",
    "COALESCE",
    "IFNULL",
    "CAST",
    "AS",
    "DECIMAL",
    "DOUBLE",
    "SIGNED",
    "UNSIGNED",
    "AND",
    "OR",
    "NOT",
    "IN",
    "IS",
    "NULL",
    "TRUE",
    "FALSE",
}


@dataclass
class FormulaReconciliation:
    original_formula: str = ""
    formula: str = ""
    source_columns: List[str] = field(default_factory=list)
    available_source_columns: List[str] = field(default_factory=list)
    missing_source_columns: List[str] = field(default_factory=list)
    rewritten: bool = False
    warning: str = ""


def formula_columns(formula: str, known_columns: Set[str] | set) -> List[str]:
    if not formula:
        return []
    found: List[str] = []
    for token in re.findall(r"`([^`]+)`|\b([A-Za-z_][A-Za-z0-9_]*)\b", formula):
        name = token[0] or token[1]
        if name.upper() in FORMULA_ALLOWED_TOKENS:
            continue
        if name in known_columns and name not in found:
            found.append(name)
    return found


def compile_metric_formula(formula: str, columns: Set[str] | set) -> str:
    text = str(formula or "").strip()
    if not text:
        return ""
    return _compile_formula_text(text, set(columns)) or _compile_formula_text(_prune_formula_for_schema(text, set(columns)), set(columns))


def reconcile_metric_formula_for_schema(
    formula: str,
    source_columns: List[str],
    available_columns: Set[str] | set,
    metric_key: str = "",
    table: str = "",
) -> FormulaReconciliation:
    available = set(available_columns)
    sources = [str(column) for column in source_columns if column]
    available_sources = [column for column in sources if column in available]
    missing_sources = [column for column in sources if column not in available]
    original = str(formula or "").strip()
    compiled = compile_metric_formula(original, available)
    rewritten = bool(compiled and original and not equivalent_formula_text(compiled, original))
    warning = ""
    if compiled and missing_sources and rewritten:
        label = metric_key or "metric"
        location = " on %s" % table if table else ""
        warning = (
            "指标 %s%s 的语义公式引用了当前 live schema 不存在的字段 %s；"
            "本次只使用可用字段 %s 收敛计算。"
            % (label, location, ",".join(missing_sources), ",".join(available_sources or formula_columns(compiled, available)))
        )
    return FormulaReconciliation(
        original_formula=original,
        formula=compiled,
        source_columns=sources,
        available_source_columns=available_sources,
        missing_source_columns=missing_sources,
        rewritten=rewritten,
        warning=warning,
    )


def equivalent_formula_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", (left or "").replace("`", "")).strip().lower() == re.sub(
        r"\s+", " ", (right or "").replace("`", "")
    ).strip().lower()


def _compile_formula_text(text: str, columns: Set[str]) -> str:
    if not text:
        return ""
    lowered = text.lower()
    forbidden = [";", "--", "/*", "*/", " select ", " from ", " join ", " union ", " insert ", " update ", " delete ", " drop ", " create "]
    if any(marker in " %s " % lowered for marker in forbidden):
        return ""
    segments = re.split(r"('(?:''|[^'])*')", text)
    compiled_segments: List[str] = []
    for index, segment in enumerate(segments):
        if index % 2 == 1:
            compiled_segments.append(segment)
            continue
        tokens = re.findall(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", segment)
        for token in tokens:
            if token in columns:
                continue
            if token.upper() in FORMULA_ALLOWED_TOKENS:
                continue
            return ""

        def replace_identifier(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in columns:
                return "`%s`" % token
            return token

        compiled_segments.append(re.sub(r"`?([A-Za-z_][A-Za-z0-9_]*)`?", replace_identifier, segment))
    compiled = "".join(compiled_segments)
    try:
        parsed = sqlglot.parse_one("SELECT %s AS metric_value FROM x" % compiled, read="doris")
    except Exception:
        return ""
    parsed_columns = {column.name for column in parsed.find_all(exp.Column) if column.name}
    if not parsed_columns.issubset(columns):
        return ""
    return compiled


def _prune_formula_for_schema(formula: str, columns: Set[str]) -> str:
    if not formula:
        return ""
    try:
        select = sqlglot.parse_one("SELECT %s AS metric_value FROM x" % formula, read="doris")
    except Exception:
        return ""
    expressions = list(select.expressions or [])
    if not expressions:
        return ""
    expression = expressions[0]
    if isinstance(expression, exp.Alias):
        expression = expression.this
    pruned = _prune_expression(expression.copy(), columns)
    if not pruned:
        return ""
    return pruned.sql(dialect="doris")


def _prune_expression(expression: exp.Expression, columns: Set[str]) -> exp.Expression | None:
    for case in list(expression.find_all(exp.Case)):
        new_ifs = []
        for item in case.args.get("ifs") or []:
            predicate = item.this
            pruned_predicate = _prune_predicate(predicate, columns)
            if pruned_predicate is None:
                continue
            new_item = item.copy()
            new_item.set("this", pruned_predicate)
            new_ifs.append(new_item)
        if not new_ifs:
            return None
        case.set("ifs", new_ifs)
    return expression if _expression_columns(expression).issubset(columns) else None


def _prune_predicate(predicate: exp.Expression, columns: Set[str]) -> exp.Expression | None:
    if _expression_columns(predicate).issubset(columns):
        return predicate.copy()
    if isinstance(predicate, exp.Or):
        left = _prune_predicate(predicate.this, columns)
        right = _prune_predicate(predicate.expression, columns)
        if left and right:
            return exp.Or(this=left, expression=right)
        return left or right
    if isinstance(predicate, exp.Paren):
        inner = _prune_predicate(predicate.this, columns)
        return exp.Paren(this=inner) if inner else None
    if isinstance(predicate, exp.And):
        return None
    return None


def _expression_columns(expression: exp.Expression) -> Set[str]:
    return {column.name for column in expression.find_all(exp.Column) if column.name}
