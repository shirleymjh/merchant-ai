from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List

from merchant_ai.services.cache import stable_cache_key


def semantic_request_payload(
    *,
    topics: Iterable[Any] = (),
    metrics: Iterable[Any] = (),
    dimensions: Iterable[Any] = (),
    filters: Iterable[Any] = (),
    time_range: Any = None,
    asset_version: Any = None,
    scope: Any = None,
) -> Dict[str, Any]:
    """Return the sole cache identity for governed semantic work.

    Natural-language wording and rendered prompts are deliberately excluded.
    Callers must provide canonical semantic identifiers and absolute time/scope
    values before a result is eligible for semantic caching.
    """

    return {
        "topics": sorted(set(_canonical_strings(topics))),
        "metrics": _canonical_items(metrics),
        "dimensions": sorted(set(_canonical_strings(dimensions))),
        "filters": _canonical_items(filters),
        "timeRange": _canonical_time_range(time_range),
        "assetVersion": _canonical_mapping(asset_version),
        "scope": _canonical_mapping(scope),
    }


def semantic_request_fingerprint(**kwargs: Any) -> str:
    payload = semantic_request_payload(**kwargs)
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def semantic_request_cache_key(namespace: str, **kwargs: Any) -> str:
    return stable_cache_key(namespace, semantic_request_payload(**kwargs))


def explicit_semantic_request_fingerprint(user_prompt: str) -> str:
    """Extract an explicit fingerprint from a structured LLM request.

    Generic dynamic prompts are intentionally not cacheable. This prevents
    history, observations, rows, or repair state from becoming accidental
    business-cache identity.
    """

    try:
        payload = json.loads(str(user_prompt or ""))
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    fingerprint = str(
        payload.get("semanticRequestFingerprint")
        or payload.get("semantic_request_fingerprint")
        or ""
    ).strip()
    if re.fullmatch(r"[a-fA-F0-9]{32,128}", fingerprint):
        return fingerprint.lower()
    semantic_request = payload.get("semanticRequest") or payload.get("semantic_request")
    if not isinstance(semantic_request, dict):
        return ""
    return semantic_request_fingerprint(
        topics=semantic_request.get("topics") or [],
        metrics=semantic_request.get("metrics") or [],
        dimensions=semantic_request.get("dimensions") or [],
        filters=semantic_request.get("filters") or [],
        time_range=semantic_request.get("timeRange") or semantic_request.get("time_range") or {},
        asset_version=semantic_request.get("assetVersion") or semantic_request.get("asset_version") or {},
        scope=semantic_request.get("scope") or {},
    )


def _canonical_strings(values: Iterable[Any]) -> List[str]:
    result: List[str] = []
    for value in values or []:
        if hasattr(value, "value"):
            value = value.value
        text = re.sub(r"\s+", " ", str(value or "").strip()).lower()
        if text:
            result.append(text)
    return result


def _canonical_items(values: Iterable[Any]) -> List[Any]:
    items = [_canonical_value(value) for value in values or []]
    items = [item for item in items if item not in (None, "", [], {})]
    return sorted(items, key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True, default=str))


def _canonical_mapping(value: Any) -> Dict[str, Any]:
    normalized = _canonical_value(value)
    return normalized if isinstance(normalized, dict) else ({"value": normalized} if normalized not in (None, "") else {})


def _canonical_time_range(value: Any) -> Dict[str, Any]:
    mapping = _canonical_mapping(value)
    aliases = {
        "kind": "kind",
        "startDate": "startDate",
        "start_date": "startDate",
        "endDate": "endDate",
        "end_date": "endDate",
        "days": "days",
        "timezone": "timezone",
        "anchorPolicy": "anchorPolicy",
        "anchor_policy": "anchorPolicy",
        "explicit": "explicit",
    }
    result: Dict[str, Any] = {}
    for source, target in aliases.items():
        if source in mapping and mapping[source] not in (None, ""):
            result[target] = mapping[source]
    return result


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump(by_alias=True)
    elif hasattr(value, "value") and not isinstance(value, (str, bytes)):
        value = value.value
    if isinstance(value, dict):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if item not in (None, "", [], {})
        }
    if isinstance(value, (list, tuple, set)):
        return _canonical_items(value)
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip()).lower()
    return value
