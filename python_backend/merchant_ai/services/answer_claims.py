from __future__ import annotations

import hashlib
import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, List, Set, Tuple

from merchant_ai.models import (
    AgentRunResult,
    AnswerClaim,
    AnswerClaimVerification,
    QueryPlan,
    VerifiedFact,
)
from merchant_ai.services.answer_formatting import humanize_column_name, source_aware_metric_label


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d[\d,]*(?:\.\d+)?%?")
DATETIME_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})"
    r"[ T](?P<hour>\d{1,2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?P<fraction>\.\d{1,6})?)?"
    r"(?P<timezone>Z|[+-]\d{2}:?\d{2})?(?!\d)"
)
DATE_PATTERN = re.compile(r"(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)")
ENTITY_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9_-]{2,}\d[A-Za-z0-9_-]*\b")
FACTUAL_SENTENCE_PATTERN = re.compile(r"[^。！？!?\n]+[。！？!?]?")
SYSTEM_FORMULA_EXPLANATION_PATTERN = re.compile(r"^(?:统计说明|计算说明)[:：]", flags=re.I)
SQL_FUNCTION_PATTERN = re.compile(
    r"\b(?:sum|count|avg|min|max|coalesce|nullif|round|cast|if)\s*\(",
    flags=re.I,
)
MARKDOWN_SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")


class AnswerClaimVerifier:
    """Reject factual values that cannot be traced to executed result rows."""

    def verify(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        answer: str,
    ) -> AnswerClaimVerification:
        facts = build_verified_facts(plan, run_result)
        if run_result is not None:
            run_result.verified_facts = facts
        claims = extract_answer_claims(answer)
        allowed_numbers = supported_numbers(facts)
        allowed_entities = supported_entities(question, facts)
        checked: List[AnswerClaim] = []
        unsupported: List[AnswerClaim] = []
        for claim in claims:
            fact_ids, reasons = support_for_claim(claim, question, facts, allowed_numbers, allowed_entities)
            verified = claim.model_copy(
                update={
                    "fact_ids": fact_ids,
                    "supported": not reasons,
                    "reasons": reasons,
                }
            )
            checked.append(verified)
            if reasons:
                unsupported.append(verified)
        return AnswerClaimVerification(
            passed=not unsupported,
            fact_count=len(facts),
            claims=checked,
            unsupported_claims=unsupported,
        )


def build_verified_facts(plan: QueryPlan, run_result: AgentRunResult | None) -> List[VerifiedFact]:
    if not run_result:
        return []
    intent_by_task = {intent.plan_task_id: intent for intent in plan.intents if intent.plan_task_id}
    singleton_intent = plan.intents[0] if len(plan.intents) == 1 else None
    facts: List[VerifiedFact] = []
    seen: Set[Tuple[str, int, str, str]] = set()
    for task in run_result.task_results:
        if task.query_bundle.failed or not task.query_bundle.rows:
            continue
        intent = intent_by_task.get(task.task_id) or singleton_intent
        table = next(iter(task.query_bundle.tables or []), "") or str(getattr(intent, "preferred_table", "") or "")
        role = fact_result_role(getattr(intent, "answer_mode", ""), str(getattr(intent, "group_by_column", "") or ""))
        resolution = dict(getattr(intent, "metric_resolution", {}) or {})
        metric_key = str(resolution.get("metricKey") or getattr(intent, "metric_name", "") or "")
        metric_label = str(
            resolution.get("displayName")
            or source_aware_metric_label(
                metric_key,
                str(getattr(intent, "preferred_table", "") or ""),
                str(getattr(intent, "category", "") or ""),
            )
            or metric_key
        )
        for row_index, row in enumerate(task.query_bundle.rows[:200]):
            for column, value in row.items():
                if value in (None, "") or str(column).startswith("__"):
                    continue
                key = (task.task_id, row_index, str(column), stable_value(value))
                if key in seen:
                    continue
                seen.add(key)
                label = metric_label if metric_key and str(column) == metric_key else str(column)
                facts.append(
                    VerifiedFact(
                        fact_id=fact_id(task.task_id, row_index, str(column)),
                        task_id=task.task_id,
                        table=table,
                        row_index=row_index,
                        column=str(column),
                        label=label,
                        value=value,
                        value_type=fact_value_type(str(column), value),
                        result_role=role,
                    )
                )
    if facts or not run_result.merged_query_bundle.rows:
        return facts
    for row_index, row in enumerate(run_result.merged_query_bundle.rows[:200]):
        for column, value in row.items():
            if value in (None, "") or str(column).startswith("__"):
                continue
            facts.append(
                VerifiedFact(
                    fact_id=fact_id("merged", row_index, str(column)),
                    task_id="merged",
                    table=next(iter(run_result.merged_query_bundle.tables or []), ""),
                    row_index=row_index,
                    column=str(column),
                    label=str(column),
                    value=value,
                    value_type=fact_value_type(str(column), value),
                )
            )
    return facts


def extract_answer_claims(answer: str) -> List[AnswerClaim]:
    body = factual_answer_body(answer)
    claims: List[AnswerClaim] = []
    prose_lines: List[str] = []
    lines = body.splitlines()
    line_index = 0
    while line_index < len(lines):
        headers = markdown_table_cells(lines[line_index])
        separator = markdown_table_cells(lines[line_index + 1]) if line_index + 1 < len(lines) else []
        if headers and separator and markdown_separator_row(separator):
            append_prose_claims("\n".join(prose_lines), claims)
            prose_lines = []
            line_index += 2
            row_position = 1
            while line_index < len(lines):
                cells = markdown_table_cells(lines[line_index])
                if not cells or markdown_separator_row(cells):
                    break
                claim = markdown_table_claim(headers, cells, row_position)
                if claim is not None:
                    claims.append(claim)
                row_position += 1
                line_index += 1
            continue
        prose_lines.append(lines[line_index])
        line_index += 1
    append_prose_claims("\n".join(prose_lines), claims)
    return claims


def append_prose_claims(text: str, claims: List[AnswerClaim]) -> None:
    for match in FACTUAL_SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip(" -*\t\r\n")
        if not sentence:
            continue
        numbers = extract_numeric_tokens(sentence)
        entities = [item for item in ENTITY_PATTERN.findall(sentence) if is_business_entity_token(item)]
        entities = dedupe_texts(entities)
        if not numbers and not entities:
            continue
        claims.append(AnswerClaim(text=sentence, numeric_values=numbers, entity_values=entities))


def markdown_table_cells(line: str) -> List[str]:
    text = str(line or "").strip()
    if not text.startswith("|") or text.count("|") < 2:
        return []
    if text.endswith("|"):
        text = text[1:-1]
    else:
        text = text[1:]
    return [cell.replace(r"\|", "|").strip() for cell in re.split(r"(?<!\\)\|", text)]


def markdown_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(MARKDOWN_SEPARATOR_CELL_PATTERN.fullmatch(cell.strip()) for cell in cells)


def markdown_table_claim(headers: List[str], cells: List[str], row_position: int) -> AnswerClaim | None:
    pairs: List[str] = []
    numbers: List[str] = []
    entities: List[str] = []
    for column_index, cell in enumerate(cells):
        value = str(cell or "").strip()
        if not value:
            continue
        header = str(headers[column_index] if column_index < len(headers) else "列%s" % (column_index + 1)).strip()
        pairs.append("%s：%s" % (header, value))
        if not deterministic_position_cell(header, value, row_position):
            numbers.extend(extract_numeric_tokens(value))
        entities.extend(item for item in ENTITY_PATTERN.findall(value) if is_business_entity_token(item))
    numeric_values = dedupe_texts(numbers)
    entity_values = dedupe_texts(entities)
    if not numeric_values and not entity_values:
        return None
    return AnswerClaim(
        text="；".join(pairs),
        numeric_values=numeric_values,
        entity_values=entity_values,
    )


def deterministic_position_cell(header: str, value: str, row_position: int) -> bool:
    normalized_header = re.sub(r"[\s_.#：:（）()\-]+", "", str(header or "").strip().lower())
    if normalized_header not in {"排名", "排行", "名次", "序号", "行号", "rank", "ranking", "no", "number", "index"}:
        return False
    number = decimal_token(value)
    return number is not None and number == number.to_integral_value() and number == Decimal(row_position)


def extract_numeric_tokens(text: str) -> List[str]:
    source = str(text or "")
    temporal_matches = list(DATETIME_PATTERN.finditer(source))
    temporal_spans = [match.span() for match in temporal_matches]
    date_matches = [
        match
        for match in DATE_PATTERN.finditer(source)
        if not span_overlaps(match.span(), temporal_spans)
    ]
    excluded_spans = [*temporal_spans, *(match.span() for match in date_matches)]
    formula_spans = system_formula_expression_spans(source)
    values: List[Tuple[int, str]] = [
        *((match.start(), match.group(0)) for match in temporal_matches),
        *((match.start(), match.group(0)) for match in date_matches),
    ]
    for match in NUMBER_PATTERN.finditer(source):
        if span_overlaps(match.span(), excluded_spans):
            continue
        if ignorable_formula_constant(match.group(0), match.span(), formula_spans):
            continue
        values.append((match.start(), match.group(0)))
    return dedupe_texts(value for _, value in sorted(values, key=lambda item: item[0]))


def system_formula_expression_spans(text: str) -> List[Tuple[int, int]]:
    if not SYSTEM_FORMULA_EXPLANATION_PATTERN.match(str(text or "").strip()):
        return []
    spans: List[Tuple[int, int]] = []
    for match in SQL_FUNCTION_PATTERN.finditer(text):
        opening = text.find("(", match.start(), match.end())
        closing = matching_parenthesis(text, opening)
        if closing >= 0:
            spans.append((match.start(), closing + 1))
    for match in re.finditer(r"\bCASE\b.*?\bEND\b", text, flags=re.I):
        spans.append(match.span())
    return spans


def matching_parenthesis(text: str, opening: int) -> int:
    if opening < 0:
        return -1
    depth = 0
    quote = ""
    index = opening
    while index < len(text):
        char = text[index]
        if quote:
            if char == quote and (index == 0 or text[index - 1] != "\\"):
                quote = ""
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def ignorable_formula_constant(raw: str, span: Tuple[int, int], formula_spans: List[Tuple[int, int]]) -> bool:
    number = decimal_token(raw)
    return number in {Decimal("0"), Decimal("1")} and any(
        formula_start <= span[0] and span[1] <= formula_end
        for formula_start, formula_end in formula_spans
    )


def span_overlaps(span: Tuple[int, int], candidates: Iterable[Tuple[int, int]]) -> bool:
    return any(span[0] < end and start < span[1] for start, end in candidates)


def factual_answer_body(answer: str) -> str:
    kept: List[str] = []
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if re.match(r"^规则依据[:：]", line):
            break
        kept.append(raw_line)
    return "\n".join(kept)


def support_for_claim(
    claim: AnswerClaim,
    question: str,
    facts: List[VerifiedFact],
    allowed_numbers: List[Tuple[Decimal, List[str], List[str]]],
    allowed_entities: Set[str],
) -> Tuple[List[str], List[str]]:
    fact_ids: List[str] = []
    reasons: List[str] = []
    for raw in claim.numeric_values:
        matching = fact_ids_for_value(raw, facts, claim.text)
        if matching:
            fact_ids.extend(matching)
            continue
        if contextual_question_date_supported(raw, claim.text, question):
            continue
        if contextual_question_number_supported(raw, claim.text, question):
            continue
        derived_fact_ids = supported_derived_numeric_token(raw, claim.text, allowed_numbers)
        if derived_fact_ids:
            fact_ids.extend(derived_fact_ids)
            continue
        reasons.append("unsupported_value:%s" % raw)
    for raw in claim.entity_values:
        normalized = normalize_entity(raw)
        matching = [fact.fact_id for fact in facts if normalize_entity(fact.value) == normalized]
        if matching:
            fact_ids.extend(matching)
            continue
        if normalized in allowed_entities:
            continue
        reasons.append("unsupported_entity:%s" % raw)
    return dedupe_texts(fact_ids), dedupe_texts(reasons)


def supported_numbers(facts: List[VerifiedFact]) -> List[Tuple[Decimal, List[str], List[str]]]:
    grouped: Dict[Tuple[str, str], List[VerifiedFact]] = {}
    for fact in facts:
        number = decimal_value(fact.value)
        if number is None:
            continue
        grouped.setdefault((fact.task_id, fact.column), []).append(fact)
    derived: List[Tuple[Decimal, List[str], List[str]]] = []
    for fact_group in grouped.values():
        labels = fact_label_candidates(fact_group[0])
        source_fact_ids = [fact.fact_id for fact in fact_group]
        if len(fact_group) == 1:
            derived.append((Decimal("0"), labels, source_fact_ids))
            continue
        if len(fact_group) < 2:
            continue
        numbers = [decimal_value(fact.value) for fact in fact_group]
        numbers = [number for number in numbers if number is not None]
        if len(numbers) < 2:
            continue
        first, last = numbers[0], numbers[-1]
        derived.extend(
            [
                (last - first, labels, source_fact_ids),
                (abs(last - first), labels, source_fact_ids),
            ]
        )
        if first != 0:
            percent = (last - first) / abs(first) * Decimal("100")
            derived.extend(
                [
                    (percent, labels, source_fact_ids),
                    (abs(percent), labels, source_fact_ids),
                ]
            )
    return dedupe_derived_numbers(derived)


def supported_entities(question: str, facts: List[VerifiedFact]) -> Set[str]:
    values = {
        normalize_entity(item)
        for item in ENTITY_PATTERN.findall(str(question or ""))
        if is_business_entity_token(item)
    }
    for fact in facts:
        if fact.value_type == "entity":
            values.add(normalize_entity(fact.value))
    return {item for item in values if item}


def fact_ids_for_value(raw: str, facts: List[VerifiedFact], claim_text: str = "") -> List[str]:
    if DATETIME_PATTERN.fullmatch(raw):
        normalized = normalize_temporal_token(raw)
        return [
            fact.fact_id
            for fact in facts
            if normalize_temporal_token(stable_value(fact.value)) == normalized
        ]
    if DATE_PATTERN.fullmatch(raw):
        normalized = normalize_temporal_token(raw)
        return [
            fact.fact_id
            for fact in facts
            if normalize_temporal_token(stable_value(fact.value)) == normalized
        ]
    target = decimal_token(raw)
    if target is None:
        return []
    matched: List[str] = []
    is_percent = raw.endswith("%")
    for fact in facts:
        value = decimal_value(fact.value)
        if value is None:
            continue
        candidates = [value]
        if is_percent and abs(value) <= 1:
            candidates.append(value * Decimal("100"))
        if any(decimal_close(target, candidate) for candidate in candidates):
            if fact_semantically_matches_claim(fact, claim_text, facts):
                matched.append(fact.fact_id)
    return matched


def fact_semantically_matches_claim(fact: VerifiedFact, claim_text: str, facts: List[VerifiedFact]) -> bool:
    normalized_claim = normalize_label_text(claim_text)
    labels = fact_label_candidates(fact)
    if any(label and label in normalized_claim for label in labels):
        return True
    numeric_candidates = [
        item
        for item in facts
        if decimal_value(item.value) is not None and decimal_value(item.value) == decimal_value(fact.value)
    ]
    return len(numeric_candidates) == 1 and not claim_has_measure_label(normalized_claim, labels)


def fact_label_candidates(fact: VerifiedFact) -> List[str]:
    return dedupe_texts(
        [
            normalize_label_text(fact.label),
            normalize_label_text(fact.column),
            normalize_label_text(humanize_column_name(fact.column)),
        ]
    )


def claim_has_measure_label(normalized_claim: str, expected_labels: List[str]) -> bool:
    expected_tokens = {token for label in expected_labels for token in measure_label_tokens(label)}
    claim_tokens = measure_label_tokens(normalized_claim)
    return bool(claim_tokens - expected_tokens)


def measure_label_tokens(text: str) -> Set[str]:
    normalized = normalize_label_text(text)
    tokens = {
        token
        for token in [
            "gmv",
            "金额",
            "数量",
            "比例",
            "支付",
            "退款",
            "退货",
            "订单",
            "工单",
            "赔付",
            "优惠",
            "用户",
        ]
        if token in normalized
    }
    return tokens


def normalize_label_text(value: Any) -> str:
    return re.sub(r"[\s_`：:（）()\-]+", "", str(value or "").strip().lower())


def contextual_question_number_supported(raw: str, claim_text: str, question: str) -> bool:
    target = decimal_token(raw)
    if target is None:
        return False
    question_tokens = contextual_number_tokens(question)
    return target in question_tokens and numeric_occurrences_are_contextual(raw, claim_text)


def contextual_question_date_supported(raw: str, claim_text: str, question: str) -> bool:
    if not DATE_PATTERN.fullmatch(raw):
        return False
    normalized = normalize_temporal_token(raw)
    question_dates = {normalize_temporal_token(item) for item in DATE_PATTERN.findall(str(question or ""))}
    claim_dates = {normalize_temporal_token(item) for item in DATE_PATTERN.findall(str(claim_text or ""))}
    return normalized in question_dates and normalized in claim_dates


def contextual_number_tokens(text: str) -> Set[Decimal]:
    values: Set[Decimal] = set()
    for pattern in contextual_number_patterns():
        for raw in re.findall(pattern, str(text or ""), flags=re.I):
            number = decimal_token(raw)
            if number is not None:
                values.add(number)
    return values


def contextual_number_patterns() -> List[str]:
    return [
        r"(?:最近|近|过去)\s*(\d+(?:\.\d+)?)\s*(?:天|日|周|月|年)",
        r"(?:前|top)\s*(\d+(?:\.\d+)?)\s*(?:个|名|条|项|笔|天)?",
        r"(?:超过|高于|低于|不少于|不超过|阈值)\s*(\d+(?:\.\d+)?%?)",
    ]


def numeric_occurrences_are_contextual(raw: str, text: str) -> bool:
    target = decimal_token(raw)
    if target is None:
        return False
    scrubbed = DATE_PATTERN.sub(lambda match: " " * len(match.group(0)), str(text or ""))
    contextual_spans = [
        match.span()
        for pattern in contextual_number_patterns()
        for match in re.finditer(pattern, scrubbed, flags=re.I)
    ]
    target_spans = [
        match.span()
        for match in NUMBER_PATTERN.finditer(scrubbed)
        if decimal_token(match.group(0)) == target
    ]
    return bool(target_spans) and all(
        any(context_start <= start and end <= context_end for context_start, context_end in contextual_spans)
        for start, end in target_spans
    )


def is_business_entity_token(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if re.search(r"_\d+[dhwmy]$", text):
        return False
    if any(fragment in text for fragment in ["_amt", "_amount", "_cnt", "_count", "_rate", "_ratio", "_gmv"]):
        return False
    return True


def supported_derived_numeric_token(
    raw: str,
    claim_text: str,
    allowed: Iterable[Tuple[Decimal, List[str], List[str]]],
) -> List[str]:
    target = decimal_token(raw)
    if target is None:
        return []
    normalized_claim = normalize_label_text(claim_text)
    for candidate, labels, fact_ids in allowed:
        if decimal_close(target, candidate) and any(label and label in normalized_claim for label in labels):
            return fact_ids
    return []


def decimal_close(left: Decimal, right: Decimal) -> bool:
    tolerance = max(Decimal("0.005"), abs(right) * Decimal("0.000001"))
    return abs(left - right) <= tolerance


def decimal_token(value: str) -> Decimal | None:
    text = str(value or "").strip().replace(",", "").rstrip("%")
    try:
        return Decimal(text)
    except (InvalidOperation, ValueError):
        return None


def decimal_value(value: Any) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    text = str(value or "").strip().replace(",", "")
    if not text or not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        return None
    try:
        return Decimal(text)
    except InvalidOperation:
        return None


def fact_value_type(column: str, value: Any) -> str:
    text = str(column or "").lower()
    if isinstance(value, (date, datetime)) or text == "pt" or re.search(r"(?:date|time|日期|时间)$", text):
        return "date"
    if decimal_value(value) is not None and not text.endswith("_id"):
        return "number"
    if text.endswith("_id") or ENTITY_PATTERN.fullmatch(str(value or "")):
        return "entity"
    return "text"


def fact_result_role(answer_mode: Any, group_by: str) -> str:
    mode = str(getattr(answer_mode, "value", answer_mode) or "").upper()
    if group_by == "pt" and mode == "GROUP_AGG":
        return "trend_context"
    return {
        "METRIC": "summary",
        "TOPN": "ranking",
        "GROUP_AGG": "group_summary",
        "DETAIL": "detail",
        "DERIVED": "derived",
    }.get(mode, "result")


def fact_id(task_id: str, row_index: int, column: str) -> str:
    raw = "%s:%s:%s" % (task_id or "task", row_index, column)
    return "fact_%s" % hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def stable_value(value: Any) -> str:
    if hasattr(value, "isoformat") and callable(value.isoformat):
        try:
            return str(value.isoformat())
        except Exception:
            pass
    return str(value)


def normalize_temporal_token(value: Any) -> str:
    text = stable_value(value).strip()
    datetime_match = DATETIME_PATTERN.fullmatch(text)
    if datetime_match:
        fraction = str(datetime_match.group("fraction") or "").rstrip("0").rstrip(".")
        timezone = str(datetime_match.group("timezone") or "")
        if timezone == "Z":
            timezone = "+00:00"
        elif timezone and ":" not in timezone:
            timezone = "%s:%s" % (timezone[:3], timezone[3:])
        return "%04d-%02d-%02dT%02d:%02d:%02d%s%s" % (
            int(datetime_match.group("year")),
            int(datetime_match.group("month")),
            int(datetime_match.group("day")),
            int(datetime_match.group("hour")),
            int(datetime_match.group("minute")),
            int(datetime_match.group("second") or 0),
            fraction,
            timezone,
        )
    date_match = DATE_PATTERN.fullmatch(text)
    if date_match:
        year, month, day = re.split(r"[-/]", text)
        return "%04d-%02d-%02d" % (int(year), int(month), int(day))
    return text


def normalize_entity(value: Any) -> str:
    return str(value or "").strip().lower()


def dedupe_texts(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in result:
            result.append(text)
    return result


def dedupe_derived_numbers(
    values: Iterable[Tuple[Decimal, List[str], List[str]]],
) -> List[Tuple[Decimal, List[str], List[str]]]:
    result: List[Tuple[Decimal, List[str], List[str]]] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()
    for number, labels, fact_ids in values:
        key = (str(number.normalize()), tuple(sorted(labels)))
        if key in seen:
            continue
        seen.add(key)
        result.append((number, labels, fact_ids))
    return result


def dedupe_decimals(values: Iterable[Decimal]) -> List[Decimal]:
    result: List[Decimal] = []
    for value in values:
        if not any(decimal_close(value, current) for current in result):
            result.append(value)
    return result
