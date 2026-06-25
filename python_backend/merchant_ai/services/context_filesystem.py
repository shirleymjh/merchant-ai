from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict


MERCHANT_URI_SCHEME = "merchant://"


def merchant_uri_for_semantic_ref(ref_id: str = "", topic: str = "", table: str = "", kind: str = "", key: str = "") -> str:
    """Map internal semantic refs to a stable, user-independent context URI."""

    raw_ref = str(ref_id or "")
    parts = raw_ref.split(":")
    if raw_ref.startswith("semantic:"):
        if len(parts) >= 3 and parts[-1] == "manifest":
            return "merchant://topic/%s/manifest" % _slug(parts[1])
        if len(parts) >= 4 and parts[-1] in {"asset", "table"}:
            return "merchant://topic/%s/table/%s" % (_slug(parts[1]), _slug(parts[2]))
        if len(parts) >= 5 and parts[-2] == "metric":
            return "merchant://topic/%s/table/%s/metric/%s" % (_slug(parts[1]), _slug(parts[2]), _slug(parts[-1]))
        if len(parts) >= 3 and parts[-1] == "relationships":
            return "merchant://topic/%s/relationships" % _slug(parts[1])
    if table:
        suffix = "/%s" % _slug(key) if key else ""
        noun = "metric" if "METRIC" in str(kind).upper() else "table"
        return "merchant://topic/%s/%s/%s%s" % (_slug(topic or "unknown"), noun, _slug(table), suffix)
    if topic:
        return "merchant://topic/%s/%s" % (_slug(topic), _slug(kind or "asset"))
    return "merchant://semantic/%s" % _slug(raw_ref or "unknown")


def merchant_uri_for_artifact(path: str = "", namespace: str = "", run_id: str = "", thread_id: str = "") -> str:
    target = str(path or "")
    name = Path(target).name if target else "artifact"
    if namespace:
        return "merchant://artifact/%s/%s" % (_slug(namespace), _slug(name))
    if run_id:
        return "merchant://run/%s/artifact/%s" % (_slug(run_id), _slug(name))
    if thread_id:
        return "merchant://thread/%s/artifact/%s" % (_slug(thread_id), _slug(name))
    return "merchant://artifact/%s" % _slug(name)


def context_layer_for_semantic_ref(kind: str = "", path: str = "") -> str:
    text = ("%s %s" % (kind, path)).lower()
    if "manifest" in text:
        return "L0"
    if "relationship" in text or "metric" in text or "rule" in text:
        return "L1"
    if "asset" in text or "schema" in text or "/tables/" in text:
        return "L2"
    return "L1"


def add_context_uri(payload: Dict[str, Any], *, ref_id: str = "", topic: str = "", table: str = "", kind: str = "", path: str = "") -> Dict[str, Any]:
    next_payload = dict(payload or {})
    effective_ref = ref_id or str(next_payload.get("refId") or "")
    effective_topic = topic or str(next_payload.get("topic") or "")
    effective_table = table or str(next_payload.get("table") or "")
    effective_kind = kind or str(next_payload.get("kind") or "")
    effective_path = path or str(next_payload.get("path") or "")
    next_payload["merchantUri"] = merchant_uri_for_semantic_ref(
        effective_ref,
        topic=effective_topic,
        table=effective_table,
        kind=effective_kind,
    )
    next_payload["contextLayer"] = context_layer_for_semantic_ref(effective_kind, effective_path)
    return next_payload


def context_lineage_record(stage: str, source: Dict[str, Any], action: str = "") -> Dict[str, Any]:
    return {
        "stage": stage,
        "action": action,
        "merchantUri": source.get("merchantUri") or "",
        "refId": source.get("refId") or "",
        "path": source.get("path") or source.get("relativePath") or "",
        "layer": source.get("contextLayer") or "",
        "kind": source.get("kind") or source.get("namespace") or "",
        "title": source.get("title") or "",
    }


def _slug(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    text = text.strip("/")
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff/]+", "_", text)
    return text or "unknown"
