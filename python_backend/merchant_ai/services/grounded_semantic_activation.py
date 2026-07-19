from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import Field

from merchant_ai.models import APIModel


SEMANTIC_ACTIVATION_SEAL_VERSION = "grounded_semantic_activation.v1"


class GroundedSemanticActivationSeal(APIModel):
    """Server-authored identity of the exact semantic sources for a run.

    The activation digest is produced by the governed Topic asset service.
    The seal fingerprint binds that source digest to the exact Topic set and
    its monotonic session version. Execution-graph topology is deliberately
    absent: changing how queries are split must not masquerade as a semantic
    source activation.
    """

    schema_version: str = SEMANTIC_ACTIVATION_SEAL_VERSION
    version: int
    exact_topics: list[str] = Field(default_factory=list)
    topic_set_fingerprint: str
    semantic_activation_fingerprint: str
    source_fingerprint: str
    seal_fingerprint: str
    sealed_at: str


def canonical_semantic_topics(topics: Iterable[Any]) -> list[str]:
    return sorted(
        {
            str(topic or "").strip()
            for topic in topics
            if str(topic or "").strip()
        }
    )


def valid_semantic_activation_fingerprint(value: Any) -> bool:
    token = str(value or "")
    return bool(
        len(token) == 64
        and all(character in "0123456789abcdef" for character in token)
    )


def semantic_topic_set_fingerprint(topics: Iterable[Any]) -> str:
    return _stable_hash(canonical_semantic_topics(topics))


def build_semantic_activation_seal(
    *,
    topics: Iterable[Any],
    semantic_activation_fingerprint: str,
    version: int,
) -> GroundedSemanticActivationSeal:
    exact_topics = canonical_semantic_topics(topics)
    if not exact_topics:
        raise RuntimeError("SEMANTIC_ACTIVATION_TOPIC_SET_REQUIRED")
    source_fingerprint = str(
        semantic_activation_fingerprint or ""
    ).strip()
    if not valid_semantic_activation_fingerprint(source_fingerprint):
        raise RuntimeError("SEMANTIC_ACTIVATION_SOURCE_FINGERPRINT_INVALID")
    normalized_version = int(version or 0)
    if normalized_version <= 0:
        raise RuntimeError("SEMANTIC_ACTIVATION_VERSION_INVALID")
    topic_set_fingerprint = semantic_topic_set_fingerprint(exact_topics)
    sealed_at = datetime.now(timezone.utc).isoformat()
    seal_payload = {
        "schemaVersion": SEMANTIC_ACTIVATION_SEAL_VERSION,
        "version": normalized_version,
        "exactTopics": exact_topics,
        "topicSetFingerprint": topic_set_fingerprint,
        "semanticActivationFingerprint": source_fingerprint,
        "sourceFingerprint": source_fingerprint,
        "sealedAt": sealed_at,
    }
    return GroundedSemanticActivationSeal(
        version=normalized_version,
        exact_topics=exact_topics,
        topic_set_fingerprint=topic_set_fingerprint,
        semantic_activation_fingerprint=source_fingerprint,
        source_fingerprint=source_fingerprint,
        seal_fingerprint=_stable_hash(seal_payload),
        sealed_at=sealed_at,
    )


def semantic_activation_seal_valid(
    seal: GroundedSemanticActivationSeal,
) -> bool:
    if not isinstance(seal, GroundedSemanticActivationSeal):
        return False
    exact_topics = canonical_semantic_topics(seal.exact_topics)
    if (
        seal.version <= 0
        or not exact_topics
        or exact_topics != seal.exact_topics
        or not seal.sealed_at
        or not valid_semantic_activation_fingerprint(
            seal.semantic_activation_fingerprint
        )
    ):
        return False
    topic_set_fingerprint = semantic_topic_set_fingerprint(exact_topics)
    expected_fingerprint = _stable_hash(
        {
            "schemaVersion": SEMANTIC_ACTIVATION_SEAL_VERSION,
            "version": seal.version,
            "exactTopics": exact_topics,
            "topicSetFingerprint": topic_set_fingerprint,
            "semanticActivationFingerprint": (
                seal.semantic_activation_fingerprint
            ),
            "sourceFingerprint": (
                seal.semantic_activation_fingerprint
            ),
            "sealedAt": seal.sealed_at,
        }
    )
    return bool(
        seal.schema_version == SEMANTIC_ACTIVATION_SEAL_VERSION
        and seal.topic_set_fingerprint
        == topic_set_fingerprint
        and seal.source_fingerprint
        == seal.semantic_activation_fingerprint
        and seal.seal_fingerprint == expected_fingerprint
    )


def _stable_hash(value: Any) -> str:
    digest = hashlib.sha256()
    encoder = json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    for chunk in encoder.iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()
