from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Dict, Iterable, List

from merchant_ai.models import KnowledgeRequest


def normalize_knowledge_request_text(value: Any) -> str:
    """Normalize semantic text without retaining presentation-only differences."""

    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized.strip()).casefold()


def canonical_knowledge_request_payload(request: KnowledgeRequest) -> Dict[str, Any]:
    """Return the retrieval-semantic identity of a KnowledgeRequest.

    ``reason``, ``round`` and ``request_key`` are deliberately absent: they are
    explanation/lineage fields and must not create a new retrieval obligation.
    ``expected_refs`` is a set for identity purposes, so ordering and duplicates
    from model output are not semantic changes.
    """

    request_type = getattr(request.type, "value", request.type)
    expected_refs = sorted(
        {
            normalized
            for value in request.expected_refs or []
            if (normalized := normalize_knowledge_request_text(value))
        }
    )
    return {
        "type": normalize_knowledge_request_text(request_type),
        "query": normalize_knowledge_request_text(request.query),
        "neededForTaskId": normalize_knowledge_request_text(request.needed_for_task_id),
        "sourcePhrase": normalize_knowledge_request_text(request.source_phrase),
        "expectedRefs": expected_refs,
    }


def knowledge_request_identity(request: KnowledgeRequest) -> str:
    payload = canonical_knowledge_request_payload(request)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dedupe_knowledge_requests(items: Iterable[KnowledgeRequest]) -> List[KnowledgeRequest]:
    deduped: List[KnowledgeRequest] = []
    seen: set[str] = set()
    for item in items:
        identity = knowledge_request_identity(item)
        if identity in seen:
            continue
        seen.add(identity)
        deduped.append(item)
    return deduped
