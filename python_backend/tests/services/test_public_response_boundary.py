from __future__ import annotations

from merchant_ai.models import ChatResponse
from merchant_ai.services.runs import public_response_payload


def test_public_chat_payload_never_contains_internal_debug_trace() -> None:
    response = ChatResponse(
        answer="verified answer",
        debug_trace={
            "internalPrompt": "do not disclose",
            "sqlCandidate": "SELECT secret_column FROM internal_table",
        },
    )

    payload = public_response_payload(response)

    assert payload["answer"] == "verified answer"
    assert "debugTrace" not in payload
    assert "debug_trace" not in payload
