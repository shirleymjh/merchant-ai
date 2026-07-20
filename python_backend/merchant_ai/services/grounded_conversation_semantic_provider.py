from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from pydantic import ConfigDict, model_validator

from merchant_ai.models import APIModel
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationDownstreamOperation,
    ConversationReferenceType,
    ConversationSemanticProviderOutput,
    ConversationSemanticResolverRequest,
)


class ConversationSemanticModelDecision(APIModel):
    """Narrow model output; server code supplies every authority binding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    complete: bool
    reference_detected: bool
    ambiguous: bool = False
    selected_artifact_id: str = ""
    referent_type: ConversationReferenceType = ConversationReferenceType.NONE
    downstream_operation: ConversationDownstreamOperation = (
        ConversationDownstreamOperation.UNSPECIFIED
    )
    population_required: bool = False
    complete_membership_required: bool = False
    current_turn_replaces_time_scope: bool = False
    reference_phrases: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_structure(self) -> "ConversationSemanticModelDecision":
        normalized = tuple(
            str(value or "").strip() for value in self.reference_phrases
        )
        if any(not value for value in normalized):
            raise ValueError("reference_phrases must not contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError(
                "reference_phrases must not contain duplicate values"
            )
        return self


class StructuredConversationSemanticProvider:
    """Isolated, zero-tool model adapter for cross-turn reference semantics."""

    _SYSTEM_PROMPT = """You are an isolated semantic reference resolver.
You receive one current user utterance and a server-retained list of prior result descriptors.
Decide whether the utterance semantically refers to prior context. If it does, identify exactly one retained artifact or mark the reference ambiguous, classify the referent type and downstream operation, and state whether complete row-population membership is required. Decide whether an explicit scope in the current utterance replaces the retained time scope.
The downstream operation describes only what the user wants to do with a referenced prior result; it does not describe the standalone operation in the current utterance. When reference_detected is false, return the non-reference defaults: ambiguous=false, selected_artifact_id="", referent_type=NONE, downstream_operation=UNSPECIFIED, population_required=false, complete_membership_required=false, current_turn_replaces_time_scope=false, and reference_phrases=[].
Use only supplied artifact IDs. Do not infer tables, formulas, SQL, row values, metrics, or business rules. Do not treat a shared time label as proof that two artifacts are the same population. Do not treat preview membership as complete. Return only the strict structured decision schema."""

    def __init__(self, model: Any, *, authority_fingerprint: str) -> None:
        authority = str(authority_fingerprint or "").strip()
        if model is None:
            raise ValueError("structured semantic model is required")
        if not authority:
            raise ValueError("authority_fingerprint is required")
        self.model = model
        self._authority_fingerprint = authority

    @property
    def authority_fingerprint(self) -> str:
        return self._authority_fingerprint

    def resolve_conversation_reference(
        self,
        request: ConversationSemanticResolverRequest,
        *,
        timeout_seconds: float,
    ) -> ConversationSemanticProviderOutput:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        bind = getattr(self.model, "with_structured_output", None)
        if not callable(bind):
            raise RuntimeError("STRUCTURED_SEMANTIC_MODEL_REQUIRED")
        structured_model = bind(
            ConversationSemanticModelDecision,
            method="json_schema",
            strict=True,
        )
        payload = {
            "protocolVersion": request.protocol_version,
            "question": request.question,
            "candidateSetFingerprint": request.candidate_set_fingerprint,
            "candidates": [
                candidate.model_dump(by_alias=True, mode="json")
                for candidate in request.candidates
            ],
        }
        raw = structured_model.invoke(
            [
                ("system", self._SYSTEM_PROMPT),
                (
                    "human",
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ]
        )
        decision = self._canonicalize_decision(
            self._parse_decision(raw)
        )
        return ConversationSemanticProviderOutput(
            request_fingerprint=request.request_fingerprint,
            question_fingerprint=request.question_fingerprint,
            candidate_set_fingerprint=request.candidate_set_fingerprint,
            **decision.model_dump(by_alias=False, mode="python"),
        )

    @staticmethod
    def _canonicalize_decision(
        decision: ConversationSemanticModelDecision,
    ) -> ConversationSemanticModelDecision:
        """Remove conditionally irrelevant fields from a non-reference result.

        The model owns the semantic decision whether a cross-turn reference is
        present. Once it says there is no reference, reference-only fields have
        no meaning and are deterministically projected to their protocol
        defaults before crossing the independent review boundary.
        """

        if decision.reference_detected:
            return decision
        return decision.model_copy(
            update={
                "ambiguous": False,
                "selected_artifact_id": "",
                "referent_type": ConversationReferenceType.NONE,
                "downstream_operation": (
                    ConversationDownstreamOperation.UNSPECIFIED
                ),
                "population_required": False,
                "complete_membership_required": False,
                "current_turn_replaces_time_scope": False,
                "reference_phrases": (),
            }
        )

    @staticmethod
    def _parse_decision(value: Any) -> ConversationSemanticModelDecision:
        if isinstance(value, ConversationSemanticModelDecision):
            return value
        if isinstance(value, Mapping):
            return ConversationSemanticModelDecision.model_validate(
                dict(value)
            )
        content: Optional[Any] = getattr(value, "content", None)
        if isinstance(content, Mapping):
            return ConversationSemanticModelDecision.model_validate(
                dict(content)
            )
        raise TypeError("structured semantic model returned an invalid value")
