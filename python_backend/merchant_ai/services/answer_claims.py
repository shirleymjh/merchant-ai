from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.text_parsing import (
    contains_any_literal,
    split_on_characters,
)
from merchant_ai.services.time_semantics import declared_time_column_for_intent



@dataclass(frozen=True)
class _TextMatch:
    source: str
    start_index: int
    end_index: int
    groups: Dict[str, str] | None = None

    def group(self, key: int | str = 0) -> str:
        if key == 0:
            return self.source[self.start_index : self.end_index]
        return str((self.groups or {}).get(str(key), ""))

    def start(self) -> int:
        return self.start_index

    def end(self) -> int:
        return self.end_index

    def span(self) -> Tuple[int, int]:
        return self.start_index, self.end_index


class _DeterministicScanner:
    def __init__(self, finder: Any) -> None:
        self.finder = finder

    def finditer(self, value: Any) -> List[_TextMatch]:
        return list(self.finder(str(value or "")))

    def findall(self, value: Any) -> List[str]:
        return [match.group(0) for match in self.finditer(value)]

    def fullmatch(self, value: Any) -> _TextMatch | None:
        text = str(value or "")
        matches = self.finditer(text)
        if len(matches) != 1:
            return None
        match = matches[0]
        return match if match.span() == (0, len(text)) else None


def _ascii_digit_end(
    text: str,
    start: int,
    *,
    minimum: int,
    maximum: int,
) -> int:
    cursor = start
    while (
        cursor < len(text)
        and cursor - start < maximum
        and text[cursor].isascii()
        and text[cursor].isdigit()
    ):
        cursor += 1
    return cursor if cursor - start >= minimum else -1


def _date_match_at(text: str, start: int) -> _TextMatch | None:
    if start > 0 and text[start - 1].isdigit():
        return None
    year_end = _ascii_digit_end(text, start, minimum=4, maximum=4)
    if year_end < 0 or year_end >= len(text) or text[year_end] not in {"-", "/"}:
        return None
    month_start = year_end + 1
    month_end = _ascii_digit_end(text, month_start, minimum=1, maximum=2)
    if month_end < 0 or month_end >= len(text) or text[month_end] not in {"-", "/"}:
        return None
    day_start = month_end + 1
    day_end = _ascii_digit_end(text, day_start, minimum=1, maximum=2)
    if day_end < 0 or (day_end < len(text) and text[day_end].isdigit()):
        return None
    return _TextMatch(
        text,
        start,
        day_end,
        {
            "year": text[start:year_end],
            "month": text[month_start:month_end],
            "day": text[day_start:day_end],
        },
    )


def _date_matches(text: str) -> Iterable[_TextMatch]:
    cursor = 0
    while cursor < len(text):
        match = _date_match_at(text, cursor)
        if match is None:
            cursor += 1
            continue
        yield match
        cursor = match.end()


def _datetime_matches(text: str) -> Iterable[_TextMatch]:
    for date_match in _date_matches(text):
        cursor = date_match.end()
        if cursor >= len(text) or text[cursor] not in {" ", "T"}:
            continue
        hour_start = cursor + 1
        hour_end = _ascii_digit_end(text, hour_start, minimum=1, maximum=2)
        if hour_end < 0 or hour_end >= len(text) or text[hour_end] != ":":
            continue
        minute_start = hour_end + 1
        minute_end = _ascii_digit_end(text, minute_start, minimum=2, maximum=2)
        if minute_end < 0:
            continue
        cursor = minute_end
        second = ""
        fraction = ""
        if cursor < len(text) and text[cursor] == ":":
            second_start = cursor + 1
            second_end = _ascii_digit_end(text, second_start, minimum=2, maximum=2)
            if second_end < 0:
                continue
            second = text[second_start:second_end]
            cursor = second_end
            if cursor < len(text) and text[cursor] == ".":
                fraction_start = cursor
                fraction_end = _ascii_digit_end(
                    text,
                    cursor + 1,
                    minimum=1,
                    maximum=6,
                )
                if fraction_end < 0:
                    continue
                fraction = text[fraction_start:fraction_end]
                cursor = fraction_end
        timezone = ""
        if cursor < len(text) and text[cursor] == "Z":
            timezone = "Z"
            cursor += 1
        elif cursor < len(text) and text[cursor] in {"+", "-"}:
            zone_start = cursor
            zone_hour_end = _ascii_digit_end(
                text,
                cursor + 1,
                minimum=2,
                maximum=2,
            )
            if zone_hour_end < 0:
                continue
            cursor = zone_hour_end
            if cursor < len(text) and text[cursor] == ":":
                cursor += 1
            zone_minute_end = _ascii_digit_end(
                text,
                cursor,
                minimum=2,
                maximum=2,
            )
            if zone_minute_end < 0:
                continue
            cursor = zone_minute_end
            timezone = text[zone_start:cursor]
        if cursor < len(text) and text[cursor].isdigit():
            continue
        groups = dict(date_match.groups or {})
        groups.update(
            {
                "hour": text[hour_start:hour_end],
                "minute": text[minute_start:minute_end],
                "second": second,
                "fraction": fraction,
                "timezone": timezone,
            }
        )
        yield _TextMatch(text, date_match.start(), cursor, groups)


def _number_matches(text: str) -> Iterable[_TextMatch]:
    cursor = 0
    ascii_word = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
    while cursor < len(text):
        start = cursor
        if text[cursor] in {"+", "-"}:
            cursor += 1
        if cursor >= len(text) or not (
            text[cursor].isascii() and text[cursor].isdigit()
        ):
            cursor = start + 1
            continue
        if start > 0 and text[start - 1] in ascii_word:
            cursor = start + 1
            continue
        while cursor < len(text) and (
            (text[cursor].isascii() and text[cursor].isdigit())
            or text[cursor] == ","
        ):
            cursor += 1
        if cursor < len(text) and text[cursor] == ".":
            decimal_end = _ascii_digit_end(
                text,
                cursor + 1,
                minimum=1,
                maximum=max(1, len(text) - cursor - 1),
            )
            if decimal_end > 0:
                cursor = decimal_end
        if cursor < len(text) and text[cursor] == "%":
            cursor += 1
        yield _TextMatch(text, start, cursor)


def _entity_matches(text: str) -> Iterable[_TextMatch]:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    cursor = 0
    while cursor < len(text):
        if not (text[cursor].isascii() and text[cursor].isalpha()):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text) and text[cursor] in allowed:
            cursor += 1
        token = text[start:cursor]
        if len(token) >= 4 and any(
            character.isdigit() for character in token[3:]
        ):
            yield _TextMatch(text, start, cursor)


def _factual_sentence_matches(text: str) -> Iterable[_TextMatch]:
    separators = {"。", "！", "？", "!", "?", "\n"}
    start = 0
    cursor = 0
    while cursor < len(text):
        if text[cursor] not in separators:
            cursor += 1
            continue
        end = cursor if text[cursor] == "\n" else cursor + 1
        if end > start:
            yield _TextMatch(text, start, end)
        cursor += 1
        start = cursor
    if start < len(text):
        yield _TextMatch(text, start, len(text))


NUMBER_PATTERN = _DeterministicScanner(_number_matches)
DATETIME_PATTERN = _DeterministicScanner(_datetime_matches)
DATE_PATTERN = _DeterministicScanner(_date_matches)
ENTITY_PATTERN = _DeterministicScanner(_entity_matches)
FACTUAL_SENTENCE_PATTERN = _DeterministicScanner(
    _factual_sentence_matches
)


def _is_system_formula_explanation(value: Any) -> bool:
    text = str(value or "").strip()
    headings = load_language_policy().answer.formula_explanation_headings
    return any(
        text.casefold().startswith(prefix.casefold() + separator)
        for prefix in headings
        for separator in (":", "：")
    )


def _strip_ordered_list_prefix(value: str) -> str:
    text = str(value or "")
    cursor = 0
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    token_start = cursor
    while cursor < len(text) and (
        (text[cursor].isascii() and text[cursor].isdigit())
        or text[cursor].isnumeric()
    ):
        cursor += 1
    if cursor == token_start or cursor >= len(text) or text[cursor] not in {".", "、", ")"}:
        return text
    cursor += 1
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    return text[cursor:]


def _is_negated_assertion(value: Any) -> bool:
    text = str(value or "")
    policy = load_language_policy().answer
    if contains_any_literal(text, policy.negated_assertion_markers):
        return True
    for prefix in policy.missing_evidence_prefixes:
        start = text.find(prefix)
        while start >= 0:
            evidence = text.find(policy.evidence_noun, start + len(prefix))
            if 0 <= evidence - (start + len(prefix)) <= 24:
                return True
            start = text.find(prefix, start + 1)
    return False


def _markdown_separator_cell(value: Any) -> bool:
    text = str(value or "").strip()
    if text.startswith(":"):
        text = text[1:]
    if text.endswith(":"):
        text = text[:-1]
    return len(text) >= 3 and set(text) == {"-"}


def _split_unescaped_pipes(value: str) -> List[str]:
    output: List[str] = []
    current: List[str] = []
    escaped = False
    for character in str(value or ""):
        if character == "|" and not escaped:
            output.append("".join(current))
            current = []
            continue
        current.append(character)
        if character == "\\":
            escaped = not escaped
        else:
            escaped = False
    output.append("".join(current))
    return output


def _split_clauses(value: Any, separators: str) -> List[str]:
    return [
        item.strip()
        for item in split_on_characters(value, separators)
        if item.strip()
    ]


def _normalize_label_characters(value: Any) -> str:
    removed = {"_", "`", "：", ":", "（", "）", "(", ")", "-"}
    return "".join(
        character
        for character in str(value or "").strip().lower()
        if not character.isspace() and character not in removed
    )


def _sql_function_spans(text: str) -> List[Tuple[int, int]]:
    names = {"sum", "count", "avg", "min", "max", "coalesce", "nullif", "round", "cast", "if"}
    spans: List[Tuple[int, int]] = []
    cursor = 0
    while cursor < len(text):
        if not (
            text[cursor].isascii()
            and (text[cursor].isalpha() or text[cursor] == "_")
        ):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text) and (
            text[cursor].isascii()
            and (text[cursor].isalnum() or text[cursor] == "_")
        ):
            cursor += 1
        token = text[start:cursor].casefold()
        scan = cursor
        while scan < len(text) and text[scan].isspace():
            scan += 1
        if token in names and scan < len(text) and text[scan] == "(":
            spans.append((start, scan + 1))
    return spans


def _case_expression_spans(text: str) -> List[Tuple[int, int]]:
    tokens: List[Tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(text):
        if not (text[cursor].isascii() and text[cursor].isalpha()):
            cursor += 1
            continue
        start = cursor
        cursor += 1
        while cursor < len(text) and text[cursor].isascii() and text[cursor].isalpha():
            cursor += 1
        tokens.append((start, cursor, text[start:cursor].casefold()))
    spans: List[Tuple[int, int]] = []
    for index, (start, _, token) in enumerate(tokens):
        if token != "case":
            continue
        closing = next(
            (
                end
                for _, end, candidate in tokens[index + 1 :]
                if candidate == "end"
            ),
            -1,
        )
        if closing >= 0:
            spans.append((start, closing))
    return spans

DerivedNumber = Tuple[Decimal, List[str], List[str], str]


class AnswerClaimVerifier:
    """Reject factual values that cannot be traced to executed result rows."""

    def verify(
        self,
        question: str,
        plan: QueryPlan,
        run_result: AgentRunResult | None,
        answer: str,
        support_context: str = "",
    ) -> AnswerClaimVerification:
        contextual_question = "\n".join(
            item for item in [str(question or ""), str(support_context or "")] if item
        )
        facts = build_verified_facts(plan, run_result)
        if run_result is not None:
            run_result.verified_facts = facts
        claims = extract_answer_claims(answer, facts)
        allowed_numbers = supported_numbers(facts)
        allowed_entities = supported_entities(contextual_question, facts, plan)
        checked: List[AnswerClaim] = []
        unsupported: List[AnswerClaim] = []
        for claim in claims:
            fact_ids, reasons = support_for_claim(
                claim,
                contextual_question,
                facts,
                allowed_numbers,
                allowed_entities,
            )
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
        column_aggregation_policies = governed_column_aggregation_policies(intent, task)
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
                        aggregation_policy=str(column_aggregation_policies.get(str(column)) or ""),
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
        for value in [fact.column, fact.label, *fact.label_aliases]
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
        sentence = _strip_ordered_list_prefix(sentence).strip()
        if not sentence:
            continue
        numbers = extract_numeric_tokens(sentence)
        entities = [item for item in ENTITY_PATTERN.findall(sentence) if is_business_entity_token(item, non_entity_tokens)]
        entities = dedupe_texts(entities)
        if not numbers and not entities and not qualitative_trend_assertion(sentence, non_entity_tokens or set()):
            continue
        claims.append(AnswerClaim(text=sentence, numeric_values=numbers, entity_values=entities))


def qualitative_trend_assertion(sentence: str, known_metric_labels: Set[str]) -> bool:
    """Return whether prose asserts a direction for a governed result label.

    Direction-only conclusions previously disappeared before verification
    because they contained neither a number nor an entity.  Labels come from
    executed facts, so this does not encode any business metric or table name.
    """

    if _is_negated_assertion(sentence):
        return False
    if not direction_word(sentence):
        return False
    normalized_sentence = normalize_label_text(sentence)
    return any(label and label in normalized_sentence for label in known_metric_labels)


def markdown_table_cells(line: str) -> List[str]:
    text = str(line or "").strip()
    if not text.startswith("|") or text.count("|") < 2:
        return []
    if text.endswith("|"):
        text = text[1:-1]
    else:
        text = text[1:]
    return [
        cell.replace(r"\|", "|").strip()
        for cell in _split_unescaped_pipes(text)
    ]


def markdown_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(
        _markdown_separator_cell(cell) for cell in cells
    )


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
    normalized_header = "".join(
        character
        for character in str(header or "").strip().lower()
        if not character.isspace()
        and character not in {"_", ".", "#", "：", ":", "（", "）", "(", ")", "-"}
    )
    if normalized_header not in load_language_policy().answer.position_column_labels:
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
    if not _is_system_formula_explanation(text):
        return []
    spans: List[Tuple[int, int]] = []
    for start, end in _sql_function_spans(text):
        opening = text.find("(", start, end)
        closing = matching_parenthesis(text, opening)
        if closing >= 0:
            spans.append((start, closing + 1))
    spans.extend(_case_expression_spans(text))
    return spans


def system_contextual_number_spans(text: str) -> List[Tuple[int, int]]:
    if not _is_system_formula_explanation(text):
        return []
    return _contextual_number_spans(text)


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
    headings = load_language_policy().answer.rule_evidence_headings
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if any(line.startswith((heading + ":", heading + "：")) for heading in headings):
            break
        kept.append(raw_line)
    return "\n".join(kept)


def support_for_claim(
    claim: AnswerClaim,
    question: str,
    facts: List[VerifiedFact],
    allowed_numbers: List[DerivedNumber],
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
    if _is_negated_assertion(text):
        return "", []
    direction_markers = load_language_policy().answer.trend_direction_markers
    if not contains_any_literal(text, tuple(marker for values in direction_markers.values() for marker in values)):
        return "", []
    matched_fact_ids: List[str] = []
    trends = trend_fact_groups(facts)
    clauses = _split_clauses(text, "；;。.!！?？\n")
    for clause in clauses:
        claimed = direction_word(clause)
        if not claimed:
            continue
        normalized_clause = normalize_label_text(clause)
        matching_trends = []
        for trend in trends:
            labels = [label for label in trend["labels"] if label]
            if any(label in normalized_clause for label in labels):
                matching_trends.append(trend)
        if not matching_trends:
            # A single governed series can provide the grammatical subject for
            # a follow-on clause such as "较上期下降 5".  Multiple/no series are
            # ambiguous and therefore fail closed.
            matching_trends = trends if len(trends) == 1 else []
        if not matching_trends:
            return "unsupported_trend_direction:no_matching_series", []
        for trend in matching_trends:
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
                    "labels": fact_label_candidates(fact),
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
                "labels": dedupe_texts(
                    label
                    for item in ordered
                    for label in item.get("labels") or []
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
    compact_clause = "".join(str(clause or "").split())
    if contains_any_literal(
        compact_clause,
        load_language_policy().answer.previous_day_comparison_markers,
    ) and claim_dates:
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
    boundaries = [
        tail.find(separator)
        for separator in "；;。.!！?？\n"
        if tail.find(separator) >= 0
    ]
    return tail[: min(boundaries)] if boundaries else tail


def direction_word(text: str) -> str:
    for direction in ("flat", "up", "down"):
        if contains_any_literal(
            text,
            load_language_policy().answer.trend_direction_markers.get(direction, ()),
        ):
            return direction
    return ""


def supported_numbers(facts: List[VerifiedFact]) -> List[DerivedNumber]:
    grouped: Dict[Tuple[str, str], List[VerifiedFact]] = {}
    for fact in facts:
        number = decimal_value(fact.value)
        if number is None:
            continue
        grouped.setdefault((fact.task_id, fact.column), []).append(fact)
    derived: List[DerivedNumber] = []
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
                (last - first, labels, source_fact_ids, "window_delta"),
                (abs(last - first), labels, source_fact_ids, "window_delta"),
            ]
        )
        if first != 0:
            percent = (last - first) / abs(first) * Decimal("100")
            derived.extend(
                [
                    (percent, labels, source_fact_ids, "window_change_rate"),
                    (abs(percent), labels, source_fact_ids, "window_change_rate"),
                ]
            )
        policies = {
            str(fact.aggregation_policy or "").strip().lower()
            for fact in fact_group
            if str(fact.aggregation_policy or "").strip()
        }
        if policies == {"period_rollup"}:
            derived.append((sum(numbers, Decimal("0")), labels, source_fact_ids, "period_rollup"))
        for previous, current in zip(fact_group, fact_group[1:]):
            previous_value = decimal_value(previous.value)
            current_value = decimal_value(current.value)
            if previous_value is None or current_value is None:
                continue
            adjacent_ids = [previous.fact_id, current.fact_id]
            adjacent_delta = current_value - previous_value
            derived.extend(
                [
                    (adjacent_delta, labels, adjacent_ids, "adjacent_delta"),
                    (abs(adjacent_delta), labels, adjacent_ids, "adjacent_delta"),
                ]
            )
            if previous_value != 0:
                adjacent_percent = adjacent_delta / abs(previous_value) * Decimal("100")
                derived.extend(
                    [
                        (adjacent_percent, labels, adjacent_ids, "adjacent_change_rate"),
                        (abs(adjacent_percent), labels, adjacent_ids, "adjacent_change_rate"),
                    ]
                )
    return dedupe_derived_numbers(derived)


def supported_entities(
    question: str,
    facts: List[VerifiedFact],
    plan: QueryPlan | None = None,
) -> Set[str]:
    values = {
        normalize_entity(item)
        for item in ENTITY_PATTERN.findall(str(question or ""))
        if is_business_entity_token(item)
    }
    for fact in facts:
        if fact.value_type == "entity":
            values.add(normalize_entity(fact.value))
    for intent in list(getattr(plan, "intents", []) or []):
        for value in [
            getattr(intent, "preferred_table", ""),
            getattr(intent, "metric_name", ""),
            getattr(intent, "metric_column", ""),
            getattr(intent, "group_by_column", ""),
        ]:
            if str(value or "").strip():
                values.add(normalize_entity(value))
        resolution = dict(getattr(intent, "metric_resolution", {}) or {})
        for value in [
            resolution.get("metricKey"),
            resolution.get("ownerTable"),
            *(resolution.get("sourceColumns") or []),
        ]:
            if str(value or "").strip():
                values.add(normalize_entity(value))
        for spec in getattr(intent, "metric_specs", []) or []:
            if not isinstance(spec, dict):
                continue
            for value in [
                spec.get("metricName"),
                spec.get("ownerTable"),
                *(spec.get("sourceColumns") or []),
            ]:
                if str(value or "").strip():
                    values.add(normalize_entity(value))
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
    clauses = _split_clauses(text, "；;。.!！?？\n")
    contexts: List[str] = []
    for clause in clauses:
        segments = _split_clauses(clause, "，,")
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
    return _normalize_label_characters(value)


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
    source = str(text or "")
    spans = _contextual_number_spans(source)
    for match in NUMBER_PATTERN.finditer(source):
        if not any(
            start <= match.start() and match.end() <= end
            for start, end in spans
        ):
            continue
        number = decimal_token(match.group(0))
        if number is not None:
            values.add(number)
    return values


def _marker_ending_before(
    text: str,
    number_start: int,
    markers: Iterable[str],
    *,
    allow_assignment_separator: bool = False,
) -> Tuple[int, str] | None:
    cursor = number_start
    while cursor > 0 and text[cursor - 1].isspace():
        cursor -= 1
    if allow_assignment_separator:
        while cursor > 0 and text[cursor - 1] in {"=", ":", "："}:
            cursor -= 1
        while cursor > 0 and text[cursor - 1].isspace():
            cursor -= 1
    folded = text.casefold()
    for marker in sorted(
        {str(item) for item in markers if str(item)},
        key=len,
        reverse=True,
    ):
        start = cursor - len(marker)
        if start >= 0 and folded[start:cursor] == marker.casefold():
            return start, marker
    return None


def _contextual_number_spans(text: str) -> List[Tuple[int, int]]:
    source = str(text or "")
    language = load_language_policy()
    temporal_prefixes = language.temporal.prefixes
    temporal_units = tuple(language.temporal.units)
    ranking_prefixes = (
        *language.routing.ranking_ordinal_prefixes,
        *language.routing.ranking_top_prefixes,
    )
    threshold_markers = language.answer.threshold_markers
    principal_markers = language.answer.principal_markers
    spans: List[Tuple[int, int]] = []
    for match in NUMBER_PATTERN.finditer(source):
        after = match.end()
        while after < len(source) and source[after].isspace():
            after += 1
        temporal_marker = _marker_ending_before(
            source,
            match.start(),
            temporal_prefixes,
        )
        if temporal_marker is not None:
            unit = next(
                (
                    item
                    for item in sorted(temporal_units, key=len, reverse=True)
                    if source.startswith(item, after)
                ),
                "",
            )
            if unit:
                spans.append((temporal_marker[0], after + len(unit)))
                continue
        ranking_marker = _marker_ending_before(
            source,
            match.start(),
            ranking_prefixes,
        )
        if ranking_marker is not None:
            suffix_end = (
                after + 1
                if after < len(source) and source[after] in language.answer.ranking_counter_characters
                else match.end()
            )
            spans.append((ranking_marker[0], suffix_end))
            continue
        threshold_marker = _marker_ending_before(
            source,
            match.start(),
            threshold_markers,
        )
        if threshold_marker is not None:
            spans.append((threshold_marker[0], match.end()))
            continue
        principal_marker = _marker_ending_before(
            source,
            match.start(),
            principal_markers,
            allow_assignment_separator=True,
        )
        if principal_marker is not None:
            spans.append((principal_marker[0], match.end()))
    return list(dict.fromkeys(spans))


def numeric_occurrences_are_contextual(raw: str, text: str) -> bool:
    target = decimal_token(raw)
    if target is None:
        return False
    source = str(text or "")
    scrubbed_characters = list(source)
    for match in DATE_PATTERN.finditer(source):
        scrubbed_characters[match.start() : match.end()] = [
            " "
        ] * (match.end() - match.start())
    scrubbed = "".join(scrubbed_characters)
    contextual_spans = _contextual_number_spans(scrubbed)
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
    allowed: Iterable[DerivedNumber],
) -> List[str]:
    target = decimal_token(raw)
    if target is None:
        return []
    direct_claim = claim_asserts_direct_metric_value(claim_text)
    normalized_claim = normalize_label_text(claim_text)
    matched_ids: List[str] = []
    for candidate, labels, fact_ids, derivation in allowed:
        if direct_claim and derivation != "period_rollup":
            continue
        if decimal_close(target, candidate) and any(label and label in normalized_claim for label in labels):
            matched_ids.extend(fact_ids)
    return dedupe_texts(matched_ids)


def claim_asserts_direct_metric_value(claim_text: str) -> bool:
    text = str(claim_text or "")
    policy = load_language_policy().answer
    if contains_any_literal(
        text,
        policy.direct_value_exclusion_markers,
        case_sensitive=False,
    ):
        return False
    markers = policy.direct_value_markers
    number_spans = [match.span() for match in NUMBER_PATTERN.finditer(text)]
    return any(
        0 <= start - (index + len(marker)) <= 12
        for marker in markers
        for index in _literal_offsets(text, marker)
        for start, _ in number_spans
    )


def _literal_offsets(text: str, literal: str) -> Iterable[int]:
    cursor = 0
    while literal and cursor <= len(text) - len(literal):
        index = text.find(literal, cursor)
        if index < 0:
            return
        yield index
        cursor = index + 1


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
        group_by_column = str(getattr(intent, "group_by_column", "") or "").strip()
        group_by_name = str(getattr(intent, "group_by_name", "") or "").strip()
        if group_by_column and group_by_name:
            labels[group_by_column] = group_by_name
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
        add(
            getattr(intent, "group_by_column", ""),
            [getattr(intent, "group_by_name", "")],
        )
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


def governed_column_aggregation_policies(intent: Any, task: Any) -> Dict[str, str]:
    """Bind a result column only to an explicitly published rollup policy.

    Policies are never inferred from identifiers, tables, formulas, or values.
    Conflicting contracts deliberately leave the column ungoverned so that a
    derived period total cannot pass answer verification.
    """

    candidates: Dict[str, Set[str]] = {}

    def add(column: Any, policy: Any) -> None:
        key = str(column or "").strip()
        value = str(policy or "").strip().lower()
        if key and value:
            candidates.setdefault(key, set()).add(value)

    if intent is not None:
        resolution = dict(getattr(intent, "metric_resolution", {}) or {})
        resolution_policy = resolution.get("aggregationPolicy") or resolution.get("aggregation_policy")
        resolution_columns = dedupe_texts(
            [
                getattr(intent, "metric_name", ""),
                getattr(intent, "metric_column", ""),
                resolution.get("metricKey"),
                resolution.get("metric_key"),
            ]
        )
        source_columns = resolution.get("sourceColumns") or resolution.get("source_columns") or []
        if len(source_columns) == 1:
            resolution_columns.append(str(source_columns[0]))
        for column in resolution_columns:
            add(column, resolution_policy)

    contract = getattr(task, "node_plan_contract", None)
    specs = [
        *(getattr(intent, "metric_specs", []) or [] if intent is not None else []),
        *(getattr(contract, "metric_specs", []) or []),
    ]
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        policy = spec.get("aggregationPolicy") or spec.get("aggregation_policy")
        columns = dedupe_texts(
            [
                spec.get("metricName"),
                spec.get("metric_name"),
                spec.get("metricColumn"),
                spec.get("metric_column"),
            ]
        )
        source_columns = spec.get("sourceColumns") or spec.get("source_columns") or []
        if len(source_columns) == 1:
            columns.append(str(source_columns[0]))
        for column in columns:
            add(column, policy)

    return {
        column: next(iter(policies))
        for column, policies in candidates.items()
        if len(policies) == 1
    }


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
    unsigned = text[1:] if text.startswith(("-", "+")) else text
    integer, separator, fraction = unsigned.partition(".")
    if (
        not integer
        or not integer.isascii()
        or not integer.isdigit()
        or (
            separator
            and (
                not fraction
                or not fraction.isascii()
                or not fraction.isdigit()
            )
        )
    ):
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
        year, month, day = text.replace("/", "-").split("-")
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
    values: Iterable[DerivedNumber],
) -> List[DerivedNumber]:
    result: List[DerivedNumber] = []
    seen: Set[Tuple[str, Tuple[str, ...]]] = set()
    for number, labels, fact_ids, derivation in values:
        key = ("%s:%s" % (derivation, number.normalize()), tuple(sorted(labels)))
        if key in seen:
            continue
        seen.add(key)
        result.append((number, labels, fact_ids, derivation))
    return result


def dedupe_decimals(values: Iterable[Decimal]) -> List[Decimal]:
    result: List[Decimal] = []
    for value in values:
        if not any(decimal_close(value, current) for current in result):
            result.append(value)
    return result
