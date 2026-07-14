from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

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
    rewrite_strategy: str = ""
    rewrite_source: str = ""
    equivalence_basis: str = ""
    degraded: bool = False
    warning: str = ""


@dataclass(frozen=True)
class _FormulaCompilation:
    formula: str = ""
    rewritten: bool = False
    rewrite_strategy: str = ""
    equivalence_basis: str = ""


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
    return _compile_metric_formula_for_schema(text, set(columns)).formula


def reconcile_metric_formula_for_schema(
    formula: str,
    source_columns: List[str],
    available_columns: Set[str] | set,
    metric_key: str = "",
    table: str = "",
) -> FormulaReconciliation:
    available = set(available_columns)
    sources = [str(column) for column in source_columns if column]
    original = str(formula or "").strip()
    referenced_columns = _formula_referenced_columns(original)
    missing_sources = _dedupe([column for column in sources + referenced_columns if column not in available])
    compilation = _compile_metric_formula_for_schema(original, available)
    compiled = compilation.formula
    compiled_columns = set(formula_columns(compiled, available))
    available_sources = [column for column in sources if column in available and (not compiled_columns or column in compiled_columns)]
    warning = ""
    if original and not compiled:
        label = metric_key or "metric"
        location = " on %s" % table if table else ""
        warning = (
            "指标 %s%s 的已发布语义公式无法在当前 live schema 上完整执行；"
            "缺失字段 %s。已禁止自动改写口径，请修复并重新发布语义资产。"
            % (label, location, ",".join(missing_sources) or "unknown")
        )
    elif compilation.rewritten:
        label = metric_key or "metric"
        location = " on %s" % table if table else ""
        warning = (
            "指标 %s%s 的 live schema 缺失字段 %s；已按已发布公式中的 OR 备选分支做受限投影。"
            "该结果仅在当前 schema 投影范围内等价，属于可追踪降级，仍需修复并重新发布语义资产。"
            % (label, location, ",".join(missing_sources) or "unknown")
        )
    return FormulaReconciliation(
        original_formula=original,
        formula=compiled,
        source_columns=sources,
        available_source_columns=available_sources,
        missing_source_columns=missing_sources,
        rewritten=compilation.rewritten,
        rewrite_strategy=compilation.rewrite_strategy,
        rewrite_source="published_formula+live_schema" if compilation.rewritten else "",
        equivalence_basis=compilation.equivalence_basis,
        degraded=compilation.rewritten,
        warning=warning,
    )


def equivalent_formula_text(left: str, right: str) -> bool:
    return re.sub(r"\s+", " ", (left or "").replace("`", "")).strip().lower() == re.sub(
        r"\s+", " ", (right or "").replace("`", "")
    ).strip().lower()


def _compile_metric_formula_for_schema(text: str, columns: Set[str]) -> _FormulaCompilation:
    compiled = _compile_formula_text(text, columns)
    if compiled:
        return _FormulaCompilation(formula=compiled)
    projected = _project_missing_or_disjuncts(text, columns)
    if not projected:
        return _FormulaCompilation()
    compiled = _compile_formula_text(projected, columns)
    if not compiled:
        return _FormulaCompilation()
    return _FormulaCompilation(
        formula=compiled,
        rewritten=not equivalent_formula_text(text, compiled),
        rewrite_strategy="project_missing_or_disjuncts",
        equivalence_basis="schema_projection_of_formula_declared_or_alternatives",
    )


def _project_missing_or_disjuncts(text: str, columns: Set[str]) -> str:
    if not _formula_text_is_safe(text):
        return ""
    try:
        parsed = sqlglot.parse_one(text, read="doris")
    except Exception:
        return ""
    referenced = {column.name for column in parsed.find_all(exp.Column) if column.name}
    if not referenced - columns:
        return ""
    rewritten = False
    for case_branch in parsed.find_all(exp.If):
        condition = case_branch.this
        condition_columns = {column.name for column in condition.find_all(exp.Column) if column.name}
        if not condition_columns - columns:
            continue
        if not any(isinstance(node, exp.Or) for node in condition.walk()):
            return ""
        if not _is_semantic_alternative_or(condition, columns):
            return ""
        projected, changed = _project_boolean_condition(condition, columns)
        if projected is None or not changed:
            return ""
        case_branch.set("this", projected)
        rewritten = True
    remaining = {column.name for column in parsed.find_all(exp.Column) if column.name}
    if not rewritten or not remaining.issubset(columns):
        return ""
    return parsed.sql(dialect="doris")


def _is_semantic_alternative_or(condition: exp.Expression, columns: Set[str]) -> bool:
    leaves = _flatten_or_conditions(condition)
    if len(leaves) < 2:
        return False
    leaf_columns = [_simple_alternative_column(leaf) for leaf in leaves]
    if any(not column for column in leaf_columns):
        return False
    distinct_columns = {str(column) for column in leaf_columns if column}
    families = {_semantic_field_family(column) for column in distinct_columns}
    return bool(
        len(distinct_columns) >= 2
        and len(families) == 1
        and "" not in families
        and bool(distinct_columns - columns)
        and bool(distinct_columns & columns)
    )


def _flatten_or_conditions(node: exp.Expression) -> List[exp.Expression]:
    current = node.this if isinstance(node, exp.Paren) else node
    if isinstance(current, exp.Or):
        return _flatten_or_conditions(current.this) + _flatten_or_conditions(current.expression)
    return [current]


def _simple_alternative_column(node: exp.Expression) -> str:
    current = node.this if isinstance(node, exp.Paren) else node
    if not isinstance(current, (exp.EQ, exp.NEQ, exp.In)):
        return ""
    columns = {column.name for column in current.find_all(exp.Column) if column.name}
    return next(iter(columns)) if len(columns) == 1 else ""


def _semantic_field_family(column: str) -> str:
    text = str(column or "").strip().lower()
    match = re.fullmatch(r"(.+)_(?:code|name|label|text)$", text)
    return str(match.group(1) or "") if match else ""


def _project_boolean_condition(node: exp.Expression, columns: Set[str]) -> Tuple[Optional[exp.Expression], bool]:
    if isinstance(node, exp.Paren):
        projected, changed = _project_boolean_condition(node.this, columns)
        if projected is None:
            return None, changed
        return exp.Paren(this=projected), changed
    if isinstance(node, exp.Or):
        left, left_changed = _project_boolean_condition(node.this, columns)
        right, right_changed = _project_boolean_condition(node.expression, columns)
        if left is None and right is None:
            return None, True
        if left is None:
            return right, True
        if right is None:
            return left, True
        return exp.or_(left, right), left_changed or right_changed
    if isinstance(node, exp.And):
        left, left_changed = _project_boolean_condition(node.this, columns)
        right, right_changed = _project_boolean_condition(node.expression, columns)
        if left is None or right is None:
            return None, left_changed or right_changed
        return exp.and_(left, right), left_changed or right_changed
    referenced = {column.name for column in node.find_all(exp.Column) if column.name}
    if referenced - columns:
        return None, True
    return node.copy(), False


def _formula_referenced_columns(text: str) -> List[str]:
    if not text or not _formula_text_is_safe(text):
        return []
    try:
        parsed = sqlglot.parse_one(text, read="doris")
    except Exception:
        return []
    return _dedupe([column.name for column in parsed.find_all(exp.Column) if column.name])


def _dedupe(values: List[str]) -> List[str]:
    return list(dict.fromkeys(value for value in values if value))


def _formula_text_is_safe(text: str) -> bool:
    lowered = str(text or "").lower()
    forbidden = [
        ";",
        "--",
        "/*",
        "*/",
        " select ",
        " from ",
        " join ",
        " union ",
        " insert ",
        " update ",
        " delete ",
        " drop ",
        " create ",
    ]
    return not any(marker in " %s " % lowered for marker in forbidden)


def _compile_formula_text(text: str, columns: Set[str]) -> str:
    if not text:
        return ""
    if not _formula_text_is_safe(text):
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
