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
from merchant_ai.services.answer_formatting import source_aware_metric_label
from merchant_ai.services.query import query_bundle_complete_rows
from merchant_ai.services.time_semantics import declared_time_column_for_intent


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
        claims = extract_answer_claims(answer, facts)
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
        task_rows, _ = query_bundle_complete_rows(task.query_bundle)
        if task.query_bundle.failed or not task_rows:
            continue
        intent = intent_by_task.get(task.task_id) or singleton_intent
        table = next(iter(task.query_bundle.tables or []), "") or str(getattr(intent, "preferred_table", "") or "")
        time_column = task_time_column(intent, task)
        role = fact_result_role(
            getattr(intent, "answer_mode", ""),
            str(getattr(intent, "group_by_column", "") or ""),
            time_column,
        )
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
        governed_columns = governed_fact_columns(intent, task)
        column_labels = governed_column_labels(intent, task)
        column_label_aliases = governed_column_label_aliases(intent, task)
        for row_index, row in enumerate(task_rows[:200]):
            for column, value in row.items():
                if value in (None, "") or str(column).startswith("__"):
                    continue
                if decimal_value(value) is not None and governed_columns and str(column) not in governed_columns:
                    continue
                key = (task.task_id, row_index, str(column), stable_value(value))
                if key in seen:
                    continue
                seen.add(key)
                label = (
                    metric_label
                    if metric_key and str(column) == metric_key
                    else str(column_labels.get(str(column)) or column)
                )
                aliases = dedupe_texts(
                    [
                        label,
                        *column_label_aliases.get(str(column), []),
                    ]
                )
                facts.append(
                    VerifiedFact(
                        fact_id=fact_id(task.task_id, row_index, str(column)),
                        task_id=task.task_id,
                        table=table,
                        row_index=row_index,
                        column=str(column),
                        label=label,
                        label_aliases=aliases,
                        value=value,
                        value_type=fact_value_type(str(column), value, {time_column} if time_column else set()),
                        result_role=role,
                    )
                )
    merged_rows, _ = query_bundle_complete_rows(run_result.merged_query_bundle)
    if facts or not merged_rows:
        return facts
    time_columns = {
        declared_time_column_for_intent(intent)
        for intent in plan.intents
        if declared_time_column_for_intent(intent)
    }
    for row_index, row in enumerate(merged_rows[:200]):
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
                    value_type=fact_value_type(str(column), value, time_columns),
                )
            )
    return facts


def extract_answer_claims(answer: str, facts: List[VerifiedFact] | None = None) -> List[AnswerClaim]:
    body = factual_answer_body(answer)
    non_entity_tokens = {
        normalize_label_text(value)
        for fact in facts or []
        if decimal_value(fact.value) is not None
        for value in [fact.column, fact.label]
        if value
    }
    claims: List[AnswerClaim] = []
    prose_lines: List[str] = []
    lines = body.splitlines()
    line_index = 0
    while line_index < len(lines):
        headers = markdown_table_cells(lines[line_index])
        separator = markdown_table_cells(lines[line_index + 1]) if line_index + 1 < len(lines) else []
        if headers and separator and markdown_separator_row(separator):
            append_prose_claims("\n".join(prose_lines), claims, non_entity_tokens)
            prose_lines = []
            line_index += 2
            row_position = 1
            while line_index < len(lines):
                cells = markdown_table_cells(lines[line_index])
                if not cells or markdown_separator_row(cells):
                    break
                claim = markdown_table_claim(headers, cells, row_position, non_entity_tokens)
                if claim is not None:
                    claims.append(claim)
                row_position += 1
                line_index += 1
            continue
        prose_lines.append(lines[line_index])
        line_index += 1
    append_prose_claims("\n".join(prose_lines), claims, non_entity_tokens)
    return claims


def append_prose_claims(text: str, claims: List[AnswerClaim], non_entity_tokens: Set[str] | None = None) -> None:
    for match in FACTUAL_SENTENCE_PATTERN.finditer(text):
        sentence = match.group(0).strip(" -*\t\r\n")
        if not sentence:
            continue
        numbers = extract_numeric_tokens(sentence)
        entities = [item for item in ENTITY_PATTERN.findall(sentence) if is_business_entity_token(item, non_entity_tokens)]
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


def markdown_table_claim(
    headers: List[str],
    cells: List[str],
    row_position: int,
    non_entity_tokens: Set[str] | None = None,
) -> AnswerClaim | None:
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
        entities.extend(item for item in ENTITY_PATTERN.findall(value) if is_business_entity_token(item, non_entity_tokens))
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
    system_contextual_spans = system_contextual_number_spans(source)
    values: List[Tuple[int, str]] = [
        *((match.start(), match.group(0)) for match in temporal_matches),
        *((match.start(), match.group(0)) for match in date_matches),
    ]
    for match in NUMBER_PATTERN.finditer(source):
        if span_overlaps(match.span(), excluded_spans):
            continue
        if ignorable_formula_constant(match.group(0), match.span(), formula_spans):
            continue
        if span_overlaps(match.span(), system_contextual_spans):
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


def system_contextual_number_spans(text: str) -> List[Tuple[int, int]]:
    if not SYSTEM_FORMULA_EXPLANATION_PATTERN.match(str(text or "").strip()):
        return []
    spans: List[Tuple[int, int]] = []
    for pattern in contextual_number_patterns():
        spans.extend(match.span() for match in re.finditer(pattern, str(text or ""), flags=re.I))
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
    direction_reason, direction_fact_ids = trend_direction_support(claim.text, facts)
    if direction_fact_ids:
        fact_ids.extend(direction_fact_ids)
    if direction_reason:
        reasons.append(direction_reason)
    for raw in claim.numeric_values:
        contexts = claim_contexts_for_value(raw, claim.text) or [claim.text]
        unsupported_occurrence = False
        for context in contexts:
            matching = fact_ids_for_value(raw, facts, context)
            if matching:
                fact_ids.extend(matching)
                continue
            if contextual_question_date_supported(raw, context, question):
                continue
            if contextual_question_number_supported(raw, context, question):
                continue
            derived_fact_ids = supported_derived_numeric_token(raw, context, allowed_numbers)
            if derived_fact_ids:
                fact_ids.extend(derived_fact_ids)
                continue
            unsupported_occurrence = True
        if unsupported_occurrence:
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


def trend_direction_support(claim_text: str, facts: List[VerifiedFact]) -> Tuple[str, List[str]]:
    text = str(claim_text or "")
    if not re.search(r"(上升|增加|提升|上涨|下降|减少|降低|下滑|持平)", text):
        return "", []
    matched_fact_ids: List[str] = []
    clauses = [item.strip() for item in re.split(r"[；;。.!！?？\n]+", text) if item.strip()]
    for clause in clauses:
        normalized_clause = normalize_label_text(clause)
        for trend in trend_fact_groups(facts):
            labels = [label for label in trend["labels"] if label]
            if not any(label in normalized_clause for label in labels):
                continue
            claimed = direction_word(clause)
            if not claimed:
                continue
            expected = expected_trend_direction_for_clause(trend, clause)
            if claimed != expected:
                return "unsupported_trend_direction:%s" % claimed, list(trend["fact_ids"])
            matched_fact_ids.extend(trend["fact_ids"])
    return "", dedupe_texts(matched_fact_ids)


def trend_fact_groups(facts: List[VerifiedFact]) -> List[Dict[str, Any]]:
    by_task_row: Dict[Tuple[str, int], List[VerifiedFact]] = {}
    for fact in facts:
        by_task_row.setdefault((fact.task_id, fact.row_index), []).append(fact)
    rows_by_task: Dict[str, List[Dict[str, Any]]] = {}
    for (task_id, row_index), row_facts in by_task_row.items():
        date_fact = next((fact for fact in row_facts if fact.value_type == "date"), None)
        for fact in row_facts:
            value = decimal_value(fact.value)
            if value is None or fact.value_type != "number":
                continue
            rows_by_task.setdefault("%s:%s" % (task_id, fact.column), []).append(
                {
                    "task_id": task_id,
                    "column": fact.column,
                    "label": fact.label,
                    "row_index": row_index,
                    "date": normalize_temporal_token(date_fact.value) if date_fact else "",
                    "value": value,
                    "fact_id": fact.fact_id,
                    "date_fact_id": date_fact.fact_id if date_fact else "",
                }
            )
    groups: List[Dict[str, Any]] = []
    for rows in rows_by_task.values():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda item: (str(item["date"] or ""), int(item["row_index"])))
        first, last = ordered[0], ordered[-1]
        delta = last["value"] - first["value"]
        direction = "up" if delta > 0 else "down" if delta < 0 else "flat"
        fact_ids = [str(first["fact_id"]), str(last["fact_id"])]
        for item in [first, last]:
            if item.get("date_fact_id"):
                fact_ids.append(str(item["date_fact_id"]))
        groups.append(
            {
                "labels": fact_label_candidates(
                    VerifiedFact(
                        fact_id=str(first["fact_id"]),
                        task_id=str(first["task_id"]),
                        row_index=int(first["row_index"]),
                        column=str(first["column"]),
                        label=str(first["label"]),
                        value=first["value"],
                    )
                ),
                "direction": direction,
                "fact_ids": dedupe_texts(fact_ids),
                "points": ordered,
            }
        )
    return groups


def expected_trend_direction_for_clause(trend: Dict[str, Any], clause: str) -> str:
    points = list(trend.get("points") or [])
    if not points:
        return str(trend.get("direction") or "")
    claim_dates = [normalize_temporal_token(item) for item in DATE_PATTERN.findall(str(clause or ""))]
    if re.search(r"较\s*前(?:一|1)?(?:日|天)", str(clause or "")) and claim_dates:
        target_date = claim_dates[-1]
        for index, point in enumerate(points):
            if str(point.get("date") or "") != target_date or index <= 0:
                continue
            delta = point["value"] - points[index - 1]["value"]
            return "up" if delta > 0 else "down" if delta < 0 else "flat"
    if len(claim_dates) >= 2:
        by_date = {str(point.get("date") or ""): point for point in points}
        first = by_date.get(claim_dates[0])
        last = by_date.get(claim_dates[-1])
        if first is not None and last is not None:
            delta = last["value"] - first["value"]
            return "up" if delta > 0 else "down" if delta < 0 else "flat"
    return str(trend.get("direction") or "")


def claim_segment_for_label(text: str, normalized_label: str) -> str:
    normalized_text = normalize_label_text(text)
    index = normalized_text.find(normalized_label)
    if index < 0:
        return text
    raw_index = max(0, min(len(text), index))
    tail = text[raw_index:]
    return re.split(r"[；;。.!！?？\n]", tail, maxsplit=1)[0]


def direction_word(text: str) -> str:
    if re.search(r"(持平|不变)", text):
        return "flat"
    if re.search(r"(上升|增加|提升|上涨)", text):
        return "up"
    if re.search(r"(下降|减少|降低|下滑)", text):
        return "down"
    return ""


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
        for previous, current in zip(fact_group, fact_group[1:]):
            previous_value = decimal_value(previous.value)
            current_value = decimal_value(current.value)
            if previous_value is None or current_value is None:
                continue
            adjacent_ids = [previous.fact_id, current.fact_id]
            adjacent_delta = current_value - previous_value
            derived.extend(
                [
                    (adjacent_delta, labels, adjacent_ids),
                    (abs(adjacent_delta), labels, adjacent_ids),
                ]
            )
            if previous_value != 0:
                adjacent_percent = adjacent_delta / abs(previous_value) * Decimal("100")
                derived.extend(
                    [
                        (adjacent_percent, labels, adjacent_ids),
                        (abs(adjacent_percent), labels, adjacent_ids),
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
    return dedupe_texts(matched)


def fact_semantically_matches_claim(fact: VerifiedFact, claim_text: str, facts: List[VerifiedFact]) -> bool:
    normalized_claim = normalize_label_text(claim_text)
    labels = fact_label_candidates(fact)
    if any(label and label in normalized_claim for label in labels):
        return True
    if claim_asserts_direct_metric_value(claim_text):
        return False
    numeric_candidates = [
        item
        for item in facts
        if decimal_value(item.value) is not None and decimal_value(item.value) == decimal_value(fact.value)
    ]
    return len(numeric_candidates) == 1 and not claim_has_competing_fact_label(normalized_claim, labels, facts)


def fact_label_candidates(fact: VerifiedFact) -> List[str]:
    return dedupe_texts(
        [
            normalize_label_text(fact.label),
            *(normalize_label_text(item) for item in fact.label_aliases),
            normalize_label_text(fact.column),
        ]
    )


def claim_contexts_for_value(raw: str, claim_text: str) -> List[str]:
    """Return only clauses that actually contain the numeric token being checked.

    A generated answer may place several metrics in one Chinese sentence separated
    by semicolons.  Binding against the whole sentence lets a label from metric A
    accidentally validate a value asserted for metric B.  Clause-local context
    preserves strict metric/value binding without relying on business vocabulary.
    """

    text = str(claim_text or "")
    clauses = [item.strip() for item in re.split(r"[；;。.!！?？\n]+", text) if item.strip()]
    contexts: List[str] = []
    for clause in clauses:
        segments = [item.strip() for item in re.split(r"[，,]+", clause) if item.strip()]
        for index, segment in enumerate(segments):
            occurrences = sum(
                1
                for token in extract_numeric_tokens(segment)
                if numeric_tokens_equivalent(raw, token)
            )
            if not occurrences:
                continue
            # Carry the metric label and grammatical subject from preceding
            # comma segments, while keeping repeated direct/delta occurrences
            # as separate verification contexts.
            context = "，".join(segments[: index + 1])
            contexts.extend([context] * occurrences)
    return contexts


def numeric_tokens_equivalent(left: str, right: str) -> bool:
    if DATETIME_PATTERN.fullmatch(str(left or "")) or DATE_PATTERN.fullmatch(str(left or "")):
        return normalize_temporal_token(left) == normalize_temporal_token(right)
    left_number = decimal_token(left)
    right_number = decimal_token(right)
    return left_number is not None and right_number is not None and decimal_close(left_number, right_number)


def claim_has_competing_fact_label(
    normalized_claim: str,
    expected_labels: List[str],
    facts: List[VerifiedFact],
) -> bool:
    expected = {normalize_label_text(item) for item in expected_labels if item}
    known = {
        normalize_label_text(value)
        for fact in facts
        for value in [fact.label, fact.column]
        if value
    }
    return any(label not in expected and label in normalized_claim for label in known)


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


def is_business_entity_token(value: str, non_entity_tokens: Set[str] | None = None) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if normalize_label_text(text) in (non_entity_tokens or set()):
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
    if claim_asserts_direct_metric_value(claim_text):
        return []
    normalized_claim = normalize_label_text(claim_text)
    matched_ids: List[str] = []
    for candidate, labels, fact_ids in allowed:
        if decimal_close(target, candidate) and any(label and label in normalized_claim for label in labels):
            matched_ids.extend(fact_ids)
    return dedupe_texts(matched_ids)


def claim_asserts_direct_metric_value(claim_text: str) -> bool:
    text = str(claim_text or "")
    if re.search(r"(变化|变动|差值|差额|增量|减少|增加|上升|下降|提升|降低|波动|环比|同比|涨幅|跌幅|少了|多了|change|delta|diff|increase|decrease)", text, flags=re.I):
        return False
    return bool(re.search(r"(?:为|是|达到|等于|合计|总计|当前|本次|查询范围内|[:：]).{0,12}[-+]?\d", text))


def governed_fact_columns(intent: Any, task: Any) -> Set[str]:
    columns: Set[str] = set()
    contract = getattr(task, "node_plan_contract", None)
    columns.update(str(item) for item in getattr(contract, "visible_columns", []) or [] if str(item))
    if intent is None:
        return columns
    columns.update(
        str(item)
        for item in [
            getattr(intent, "metric_column", ""),
            getattr(intent, "metric_name", ""),
            getattr(intent, "group_by_column", ""),
            getattr(intent, "filter_column", ""),
            *(getattr(intent, "output_keys", []) or []),
            *(getattr(intent, "required_evidence", []) or []),
        ]
        if str(item)
    )
    resolution = dict(getattr(intent, "metric_resolution", {}) or {})
    columns.update(str(item) for item in resolution.get("sourceColumns") or [] if str(item))
    for spec in getattr(intent, "metric_specs", []) or []:
        if not isinstance(spec, dict):
            continue
        columns.update(str(item) for item in spec.get("sourceColumns") or [] if str(item))
        if spec.get("metricName"):
            columns.add(str(spec.get("metricName")))
    return columns


def governed_column_labels(intent: Any, task: Any) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    if intent is not None:
        resolution = dict(getattr(intent, "metric_resolution", {}) or {})
        for key in ["sourceColumnLabels", "columnLabels", "outputColumnLabels"]:
            raw = resolution.get(key)
            if isinstance(raw, dict):
                labels.update({str(column): str(label) for column, label in raw.items() if column and label})
    contract = getattr(task, "node_plan_contract", None)
    for column, policy in (getattr(contract, "column_display_policy", {}) or {}).items():
        if not isinstance(policy, dict):
            continue
        label = policy.get("displayName") or policy.get("label") or policy.get("title")
        if label:
            labels[str(column)] = str(label)
    return labels


def governed_column_label_aliases(intent: Any, task: Any) -> Dict[str, List[str]]:
    aliases: Dict[str, List[str]] = {}

    def add(column: Any, values: Iterable[Any]) -> None:
        key = str(column or "").strip()
        if not key:
            return
        aliases[key] = dedupe_texts([*aliases.get(key, []), *(str(value) for value in values if str(value or "").strip())])

    if intent is not None:
        resolution = dict(getattr(intent, "metric_resolution", {}) or {})
        primary_column = str(
            getattr(intent, "metric_column", "")
            or resolution.get("metricKey")
            or getattr(intent, "metric_name", "")
            or ""
        )
        add(
            primary_column,
            [
                resolution.get("displayName"),
                resolution.get("naturalName"),
                resolution.get("description"),
                resolution.get("sourcePhrase"),
            ],
        )
        for key in ["sourceColumnLabels", "columnLabels", "outputColumnLabels"]:
            raw = resolution.get(key)
            if isinstance(raw, dict):
                for column, label in raw.items():
                    add(column, [label])

    contract = getattr(task, "node_plan_contract", None)
    specs = [
        *(getattr(intent, "metric_specs", []) or [] if intent is not None else []),
        *(getattr(contract, "metric_specs", []) or []),
    ]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        metric_column = spec.get("metricColumn") or spec.get("metric_column")
        metric_name = spec.get("metricName") or spec.get("metric_name")
        values = [
            spec.get("displayName"),
            spec.get("display_name"),
            spec.get("naturalName"),
            spec.get("natural_name"),
            spec.get("description"),
            spec.get("sourcePhrase"),
            spec.get("source_phrase"),
        ]
        # SQL may alias an expression to metricName while metricColumn remains
        # a physical source field (for example COUNT(id) AS metric_count).
        # Both governed identifiers refer to the same published result metric.
        add(metric_column, values)
        add(metric_name, values)
        source_columns = spec.get("sourceColumns") or spec.get("source_columns") or []
        if len(source_columns) == 1:
            add(source_columns[0], values)
    return aliases


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


def fact_value_type(column: str, value: Any, declared_time_columns: Set[str] | None = None) -> str:
    text = str(column or "").lower()
    temporal_value = str(value or "").strip()
    if (
        isinstance(value, (date, datetime))
        or text in {str(item or "").strip().lower() for item in (declared_time_columns or set()) if str(item or "").strip()}
        or bool(DATETIME_PATTERN.fullmatch(temporal_value))
        or bool(DATE_PATTERN.fullmatch(temporal_value))
    ):
        return "date"
    if decimal_value(value) is not None and not text.endswith("_id"):
        return "number"
    if text.endswith("_id") or ENTITY_PATTERN.fullmatch(str(value or "")):
        return "entity"
    return "text"


def fact_result_role(answer_mode: Any, group_by: str, time_column: str = "") -> str:
    mode = str(getattr(answer_mode, "value", answer_mode) or "").upper()
    if time_column and group_by == time_column and mode == "GROUP_AGG":
        return "trend_context"
    return {
        "METRIC": "summary",
        "TOPN": "ranking",
        "GROUP_AGG": "group_summary",
        "DETAIL": "detail",
        "DERIVED": "derived",
    }.get(mode, "result")


def task_time_column(intent: Any, task: Any) -> str:
    contract = getattr(task, "node_plan_contract", None)
    time_contract = getattr(contract, "time_window_contract", None) or {}
    contract_column = declared_time_column_from_contract(time_contract)
    return contract_column or declared_time_column_for_intent(intent)


def declared_time_column_from_contract(contract: Any) -> str:
    if not isinstance(contract, dict):
        return ""
    for key in ["partitionColumn", "partition_column", "timeColumn", "time_column"]:
        value = str(contract.get(key) or "").strip()
        if value:
            return value
    return ""


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
