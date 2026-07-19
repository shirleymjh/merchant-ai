from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict


MERCHANT_URI_SCHEME = "merchant://"


class ContextPathOutsideRootError(ValueError):
    """Raised when a context file path escapes its configured workspace root."""


def resolve_context_path(root: Path | str, path: Path | str) -> Path:
    """Resolve ``path`` while enforcing the current context-root boundary.

    Resolution happens before the containment check so existing parent or file
    symlinks cannot redirect a read/write outside the workspace. Absolute paths
    remain supported for internal artifact references, but only when they
    resolve beneath the active root.
    """

    try:
        resolved_root = Path(root).expanduser().resolve(strict=True)
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = resolved_root / candidate
        resolved_candidate = candidate.resolve(strict=False)
        resolved_candidate.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise ContextPathOutsideRootError("context path is outside the configured root") from exc
    return resolved_candidate


def context_path_is_within_root(root: Path | str, path: Path | str) -> bool:
    try:
        resolve_context_path(root, path)
        return True
    except ContextPathOutsideRootError:
        return False


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
        if len(parts) >= 4 and parts[-1] in {
            "detail",
            "metrics",
            "columns",
            "schema",
            "terms",
            "rules",
        }:
            return "merchant://topic/%s/table/%s/%s" % (
                _slug(parts[1]),
                _slug(parts[2]),
                _slug(parts[-1]),
            )
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
    if "relationship" in text or "metric" in text or "table_detail" in text or "/detail.json" in text:
        return "L1"
    if any(
        marker in text
        for marker in [
            "asset",
            "schema",
            "column",
            "rule",
            "term",
            "/tables/",
        ]
    ):
        return "L2"
    return "L1"


def context_depth_for_semantic_ref(kind: str = "", path: str = "") -> int:
    """Return the progressive-disclosure depth independently of token layers.

    ``contextLayer`` remains the compact L0/L1/L2 compatibility contract used
    by existing traces. ``contextDepth`` makes the actual browse order
    explicit: table list -> table detail -> columns/schema -> business rules.
    """

    text = ("%s %s" % (kind, path)).lower()
    if "manifest" in text:
        return 0
    if "table_detail" in text or "/detail.json" in text or "relationship" in text or "metric" in text:
        return 1
    if "column" in text or "schema" in text or "term" in text:
        return 2
    if "rule" in text or "asset" in text:
        return 3
    return 1


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
    next_payload["contextDepth"] = context_depth_for_semantic_ref(effective_kind, effective_path)
    return next_payload


def context_lineage_record(stage: str, source: Dict[str, Any], action: str = "") -> Dict[str, Any]:
    return {
        "stage": stage,
        "action": action,
        "merchantUri": source.get("merchantUri") or "",
        "refId": source.get("refId") or "",
        "path": source.get("path") or source.get("relativePath") or "",
        "layer": source.get("contextLayer") or "",
        "depth": source.get("contextDepth"),
        "kind": source.get("kind") or source.get("namespace") or "",
        "title": source.get("title") or "",
    }


def _slug(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    text = _collapse_character_runs(text, "/")
    text = text.strip("/")
    text = _replace_runs(text, str.isspace, "_")
    text = _replace_runs(text, lambda character: not _is_context_slug_character(character), "_")
    return text or "unknown"


def _collapse_character_runs(value: str, character: str) -> str:
    output: list[str] = []
    previous_was_target = False
    for current in value:
        if current == character:
            if not previous_was_target:
                output.append(current)
            previous_was_target = True
            continue
        output.append(current)
        previous_was_target = False
    return "".join(output)


def _replace_runs(value: str, should_replace: Callable[[str], bool], replacement: str) -> str:
    output: list[str] = []
    replacing = False
    for character in value:
        if should_replace(character):
            if not replacing:
                output.append(replacement)
            replacing = True
            continue
        output.append(character)
        replacing = False
    return "".join(output)


def _is_context_slug_character(character: str) -> bool:
    return (
        "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or character in "_.-/"
        or "\u4e00" <= character <= "\u9fff"
    )
