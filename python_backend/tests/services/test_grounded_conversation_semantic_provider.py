from __future__ import annotations

import json

import pytest

from merchant_ai.services.grounded_conversation_semantic_provider import (
    ConversationSemanticModelDecision,
    StructuredConversationSemanticProvider,
)
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationReferentCandidate,
    build_conversation_semantic_resolver_request,
)


class _StructuredModel:
    def __init__(self, output) -> None:
        self.output = output
        self.schema = None
        self.method = ""
        self.strict = False
        self.messages = []

    def with_structured_output(self, schema, *, method: str, strict: bool):
        self.schema = schema
        self.method = method
        self.strict = strict
        return self

    def invoke(self, messages):
        self.messages = list(messages)
        return self.output


def _request():
    return build_conversation_semantic_resolver_request(
        "continue from the retained result",
        (
            ConversationReferentCandidate(
                artifact_id="artifact-a",
                contract_fingerprint="contract-a",
                sql_fingerprint="sql-a",
                query_shape="DETAIL",
                coverage_status="ALL_ROWS",
            ),
        ),
    )


def _decision():
    return ConversationSemanticModelDecision(
        complete=True,
        reference_detected=True,
        selected_artifact_id="artifact-a",
        referent_type=ConversationReferenceType.PREDICATE_SCOPE,
        downstream_operation=ConversationDownstreamOperation.RANK,
        population_required=True,
        current_turn_replaces_time_scope=False,
        reference_phrases=("retained result",),
    )


def test_provider_uses_strict_structured_model_and_server_bindings() -> None:
    model = _StructuredModel(_decision())
    provider = StructuredConversationSemanticProvider(
        model,
        authority_fingerprint="reviewer-authority",
    )
    request = _request()

    output = provider.resolve_conversation_reference(
        request,
        timeout_seconds=2.0,
    )

    assert provider.authority_fingerprint == "reviewer-authority"
    assert model.schema is ConversationSemanticModelDecision
    assert model.method == "json_schema"
    assert model.strict is True
    assert output.request_fingerprint == request.request_fingerprint
    assert output.question_fingerprint == request.question_fingerprint
    assert output.candidate_set_fingerprint == (
        request.candidate_set_fingerprint
    )
    assert output.selected_artifact_id == "artifact-a"

    supplied = json.loads(model.messages[1][1])
    assert supplied["question"] == request.question
    assert supplied["candidates"][0]["artifactId"] == "artifact-a"
    assert "rows" not in supplied["candidates"][0]
    assert "sql" not in supplied["candidates"][0]


def test_provider_accepts_mapping_decision_but_not_free_text() -> None:
    request = _request()
    mapping_model = _StructuredModel(
        _decision().model_dump(by_alias=True, mode="json")
    )
    provider = StructuredConversationSemanticProvider(
        mapping_model,
        authority_fingerprint="reviewer-authority",
    )

    assert provider.resolve_conversation_reference(
        request,
        timeout_seconds=1.0,
    ).reference_detected

    text_provider = StructuredConversationSemanticProvider(
        _StructuredModel("unstructured output"),
        authority_fingerprint="reviewer-authority",
    )
    with pytest.raises(TypeError):
        text_provider.resolve_conversation_reference(
            request,
            timeout_seconds=1.0,
        )


def test_provider_requires_model_authority_and_positive_budget() -> None:
    with pytest.raises(ValueError):
        StructuredConversationSemanticProvider(
            None,
            authority_fingerprint="reviewer-authority",
        )
    with pytest.raises(ValueError):
        StructuredConversationSemanticProvider(
            _StructuredModel(_decision()),
            authority_fingerprint="",
        )

    provider = StructuredConversationSemanticProvider(
        _StructuredModel(_decision()),
        authority_fingerprint="reviewer-authority",
    )
    with pytest.raises(ValueError):
        provider.resolve_conversation_reference(
            _request(),
            timeout_seconds=0,
        )


def test_provider_rejects_model_without_strict_structured_interface() -> None:
    provider = StructuredConversationSemanticProvider(
        object(),
        authority_fingerprint="reviewer-authority",
    )

    with pytest.raises(RuntimeError):
        provider.resolve_conversation_reference(
            _request(),
            timeout_seconds=1.0,
        )
