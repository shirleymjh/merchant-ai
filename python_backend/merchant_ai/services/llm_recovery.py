from __future__ import annotations

from dataclasses import dataclass

from merchant_ai.services.text_parsing import ASCII_DIGITS


@dataclass(frozen=True)
class LlmFailureClassification:
    kind: str
    retryable: bool


def classify_llm_failure(error: str) -> LlmFailureClassification:
    """Classify transient transport failures without coupling to an agent stage."""

    lowered = str(error or "").strip().lower()
    if "timeout" in lowered or "timed out" in lowered:
        return LlmFailureClassification("TIMEOUT", True)
    if not lowered or "empty_response" in lowered or "empty response" in lowered:
        return LlmFailureClassification("EMPTY_RESPONSE", True)

    transient_markers = (
        "connection reset",
        "connection refused",
        "connection aborted",
        "connection closed",
        "connection error",
        "connectionerror",
        "connecterror",
        "remote disconnected",
        "temporary failure in name resolution",
        "name or service not known",
        "server disconnected",
        "service unavailable",
        "bad gateway",
        "temporarily unavailable",
    )
    if _contains_status_code(lowered, 500, 599) or any(marker in lowered for marker in transient_markers):
        return LlmFailureClassification("PROVIDER_ERROR", True)
    if (
        "provider_error" in lowered
        or _contains_status_code(lowered, 400, 499)
        or any(marker in lowered for marker in ("forbidden", "unauthorized", "invalid api key"))
    ):
        return LlmFailureClassification("PROVIDER_ERROR", False)
    return LlmFailureClassification("FAILED", False)


def _contains_status_code(text: str, minimum: int, maximum: int) -> bool:
    source = str(text or "")
    cursor = 0
    while cursor < len(source):
        if source[cursor] not in ASCII_DIGITS:
            cursor += 1
            continue
        end = cursor + 1
        while end < len(source) and source[end] in ASCII_DIGITS:
            end += 1
        token = source[cursor:end]
        if len(token) == 3 and minimum <= int(token) <= maximum:
            return True
        cursor = end
    return False


def bounded_single_retry_count(configured_retries: int) -> int:
    """Network retries are a 0/1 safety switch, never an unbounded loop."""

    return min(1, max(0, int(configured_retries or 0)))


def retry_timeout_with_answer_reserve(
    remaining_seconds: float,
    request_timeout_seconds: int,
    answer_timeout_seconds: int,
) -> int:
    """Allow a full retry only when the global run can still reserve its answer lane."""

    request_timeout = max(1, int(request_timeout_seconds or 1))
    answer_reserve = max(3, int(answer_timeout_seconds or 10) + 2)
    if float(remaining_seconds or 0.0) < request_timeout + answer_reserve:
        return 0
    return request_timeout
