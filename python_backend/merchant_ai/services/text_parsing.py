from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


ASCII_LOWER = frozenset("abcdefghijklmnopqrstuvwxyz")
ASCII_UPPER = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ASCII_LETTERS = ASCII_LOWER | ASCII_UPPER
ASCII_DIGITS = frozenset("0123456789")
ASCII_HEX = ASCII_DIGITS | frozenset("abcdefABCDEF")
ASCII_WORD = ASCII_LETTERS | ASCII_DIGITS | {"_"}


def collapse_whitespace(value: object) -> str:
    """Collapse every Unicode whitespace run without pattern matching."""

    return " ".join(str(value or "").split())


def is_ascii_hex(value: object, *, minimum: int = 1, maximum: int | None = None) -> bool:
    text = str(value or "")
    upper_bound = minimum if maximum is None else maximum
    return minimum <= len(text) <= upper_bound and all(character in ASCII_HEX for character in text)


def is_ascii_identifier(value: object) -> bool:
    text = str(value or "")
    return bool(text) and text[0] in ASCII_LETTERS | {"_"} and all(
        character in ASCII_WORD for character in text
    )


def is_ascii_word_phrase(value: object, *, extras: Iterable[str] = ()) -> bool:
    text = str(value or "")
    allowed = ASCII_WORD | frozenset(extras)
    return bool(text) and all(character in allowed for character in text)


def compact_ascii_alphanumeric(value: object, *, uppercase: bool = False) -> str:
    text = str(value or "")
    text = text.upper() if uppercase else text.casefold()
    allowed = (ASCII_UPPER if uppercase else ASCII_LOWER) | ASCII_DIGITS
    return "".join(character for character in text if character in allowed)


def replace_disallowed_runs(
    value: object,
    *,
    allowed: Callable[[str], bool],
    replacement: str = "_",
) -> str:
    """Replace each consecutive disallowed run with one literal replacement."""

    output: list[str] = []
    replacing = False
    for character in str(value or ""):
        if allowed(character):
            output.append(character)
            replacing = False
            continue
        if not replacing and replacement:
            output.append(replacement)
        replacing = True
    return "".join(output)


def safe_ascii_component(
    value: object,
    *,
    extras: Iterable[str] = ("_",),
    default: str = "",
    lowercase: bool = False,
    uppercase: bool = False,
    strip: str = "_",
) -> str:
    text = str(value or "")
    if lowercase and uppercase:
        raise ValueError("safe ASCII component cannot request both lowercase and uppercase")
    if lowercase:
        text = text.lower()
    elif uppercase:
        text = text.upper()
    allowed_characters = ASCII_LETTERS | ASCII_DIGITS | frozenset(extras)
    normalized = replace_disallowed_runs(
        text,
        allowed=lambda character: character in allowed_characters,
    ).strip(strip)
    return normalized or default


def split_on_characters(value: object, separators: Iterable[str]) -> list[str]:
    separator_set = frozenset(separators)
    output: list[str] = []
    current: list[str] = []
    for character in str(value or ""):
        if character in separator_set:
            output.append("".join(current))
            current = []
        else:
            current.append(character)
    output.append("".join(current))
    return output


def contains_any_literal(value: object, literals: Iterable[str], *, case_sensitive: bool = True) -> bool:
    text = str(value or "")
    if not case_sensitive:
        text = text.casefold()
    return any(
        bool(literal) and ((literal if case_sensitive else literal.casefold()) in text)
        for literal in literals
    )


def literal_spans(
    value: object,
    literals: Iterable[str],
    *,
    case_sensitive: bool = True,
    ascii_word_boundary: bool = False,
) -> list[tuple[int, int, str]]:
    """Find non-overlapping literals, preferring the longest at each offset."""

    original = str(value or "")
    haystack = original if case_sensitive else original.casefold()
    ordered = sorted(
        {str(item) for item in literals if str(item)},
        key=lambda item: (-len(item), item),
    )
    needles = [(item, item if case_sensitive else item.casefold()) for item in ordered]
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    while cursor < len(original):
        matched = ""
        for source, needle in needles:
            if not haystack.startswith(needle, cursor):
                continue
            end = cursor + len(source)
            if ascii_word_boundary and not _has_ascii_word_boundaries(original, cursor, end):
                continue
            matched = source
            break
        if not matched:
            cursor += 1
            continue
        end = cursor + len(matched)
        spans.append((cursor, end, original[cursor:end]))
        cursor = end
    return spans


def iter_ascii_digit_spans(value: object) -> Iterator[tuple[int, int, str]]:
    text = str(value or "")
    cursor = 0
    while cursor < len(text):
        if text[cursor] not in ASCII_DIGITS:
            cursor += 1
            continue
        start = cursor
        while cursor < len(text) and text[cursor] in ASCII_DIGITS:
            cursor += 1
        yield start, cursor, text[start:cursor]


def parse_prefixed_reference(
    value: object,
    *,
    prefix: str,
    separator: str,
    part_count: int,
) -> tuple[str, ...] | None:
    text = str(value or "").strip()
    if not text.startswith(prefix):
        return None
    parts = text[len(prefix) :].split(separator)
    if len(parts) != part_count or any(not part for part in parts):
        return None
    return tuple(parts)


def exact_path_segments(value: object, *, prefix: str = "") -> tuple[str, ...] | None:
    text = str(value or "").strip("/")
    segments = tuple(text.split("/")) if text else ()
    if any(not segment or segment in {".", ".."} for segment in segments):
        return None
    if prefix and (not segments or segments[0] != prefix):
        return None
    return segments


def separator_contains_only(value: object, allowed_characters: Iterable[str]) -> bool:
    text = str(value or "")
    allowed = frozenset(allowed_characters)
    return bool(text) and all(character.isspace() or character in allowed for character in text)


def leading_iso_date_parts(value: object) -> tuple[str, str, str] | None:
    """Parse a leading YYYY-M-D portion while allowing a timestamp suffix."""

    text = str(value or "").strip()
    if len(text) < 8 or len(text) < 5 or text[4] != "-":
        return None
    year = text[:4]
    if len(year) != 4 or any(character not in ASCII_DIGITS for character in year):
        return None
    cursor = 5
    month_start = cursor
    while cursor < len(text) and text[cursor] in ASCII_DIGITS and cursor - month_start < 2:
        cursor += 1
    month = text[month_start:cursor]
    if not month or cursor >= len(text) or text[cursor] != "-":
        return None
    cursor += 1
    day_start = cursor
    while cursor < len(text) and text[cursor] in ASCII_DIGITS and cursor - day_start < 2:
        cursor += 1
    day = text[day_start:cursor]
    if not day or (cursor < len(text) and text[cursor] in ASCII_DIGITS):
        return None
    return year, month, day


def _has_ascii_word_boundaries(value: str, start: int, end: int) -> bool:
    left_ok = start == 0 or value[start - 1] not in ASCII_WORD
    right_ok = end == len(value) or value[end] not in ASCII_WORD
    return left_ok and right_ok
