from __future__ import annotations

import hashlib
import json
import re
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, List

from merchant_ai.config import Settings
from merchant_ai.services.context_filesystem import (
    ContextPathOutsideRootError,
    context_path_is_within_root,
    merchant_uri_for_artifact,
    resolve_context_path,
)


class WorkspaceArtifactStore:
    """Filesystem-backed context store for agent intermediate artifacts."""

    PATH_OUTSIDE_ROOT = "ARTIFACT_PATH_OUTSIDE_ROOT"

    def __init__(self, settings: Settings, root: Path | str | None = None):
        self.settings = settings
        default_root = Path(root) if root else settings.resolved_workspace_path / "artifacts"
        default_root.mkdir(parents=True, exist_ok=True)
        self._default_root = default_root.resolve()
        self._context_root: ContextVar[Path | None] = ContextVar("workspace_artifact_root_%x" % id(self), default=None)

    @property
    def root(self) -> Path:
        return self._context_root.get() or self._default_root

    def set_context_root(self, root: Path | str) -> None:
        target = Path(root)
        target.mkdir(parents=True, exist_ok=True)
        self._context_root.set(target.resolve())

    def with_root(self, root: Path | str) -> "WorkspaceArtifactStore":
        return WorkspaceArtifactStore(self.settings, root)

    def write_json(self, namespace: str, name: str, payload: Any, preview_chars: int | None = None) -> Dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return self.write_text(namespace, name if name.endswith(".json") else "%s.json" % name, text, preview_chars=preview_chars)

    def write_text(self, namespace: str, name: str, content: str, preview_chars: int | None = None) -> Dict[str, Any]:
        relative_path = Path(sanitize_path_part(namespace or "misc")) / sanitize_file_name(name or "artifact.txt")
        try:
            target_dir = self._resolve(str(relative_path.parent))
            target = self._resolve(str(relative_path))
        except ContextPathOutsideRootError:
            return self._path_error(str(relative_path))
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            return {"success": False, "error": "ARTIFACT_WRITE_FAILED", "path": str(relative_path)}
        # Re-resolve after directory creation so an existing symlinked parent is
        # checked immediately before the write.
        try:
            target = self._resolve(str(relative_path))
        except ContextPathOutsideRootError:
            return self._path_error(str(relative_path))
        text = str(content or "")
        try:
            target.write_text(text, encoding="utf-8")
        except OSError:
            return {"success": False, "error": "ARTIFACT_WRITE_FAILED", "path": str(relative_path)}
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        preview_limit = max(0, int(preview_chars if preview_chars is not None else self.settings.context_file_inline_max_chars))
        return {
            "success": True,
            "path": str(target),
            "relativePath": str(target.relative_to(self.root)),
            "merchantUri": merchant_uri_for_artifact(str(target.relative_to(self.root)), namespace=namespace or "misc"),
            "bytes": len(text.encode("utf-8")),
            "estimatedChars": len(text),
            "sha256": digest,
            "preview": text[:preview_limit],
            "truncated": len(text) > preview_limit,
        }

    def read(self, path: str, offset: int = 0, max_chars: int | None = None) -> Dict[str, Any]:
        try:
            target = self._resolve(path)
        except ContextPathOutsideRootError:
            return self._path_error(path)
        if not target.exists() or not target.is_file():
            return {"success": False, "error": "ARTIFACT_NOT_FOUND", "path": path}
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            return {"success": False, "error": "ARTIFACT_READ_FAILED", "path": path}
        start = max(0, int(offset or 0))
        limit = max(1, int(max_chars or self.settings.context_file_inline_max_chars))
        end = min(len(text), start + limit)
        return {
            "success": True,
            "path": str(target),
            "relativePath": str(target.relative_to(self.root)) if self._is_under_root(target) else str(target),
            "merchantUri": merchant_uri_for_artifact(str(target.relative_to(self.root)) if self._is_under_root(target) else str(target)),
            "content": text[start:end],
            "contentOffsetChars": start,
            "nextContentOffsetChars": end if end < len(text) else None,
            "truncated": end < len(text),
            "estimatedChars": len(text),
        }

    def grep(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        terms = [term.lower() for term in re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]{2,}", query or "") if term]
        if not terms:
            return []
        hits: List[Dict[str, Any]] = []
        for discovered_path in sorted(self.root.rglob("*")):
            try:
                path = self._resolve(str(discovered_path))
            except ContextPathOutsideRootError:
                continue
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lower = text.lower()
            score = sum(lower.count(term) for term in terms)
            if score <= 0:
                continue
            hits.append(
                {
                    "path": str(path),
                    "relativePath": str(path.relative_to(self.root)),
                    "merchantUri": merchant_uri_for_artifact(str(path.relative_to(self.root))),
                    "score": score,
                    "snippets": artifact_snippets(text, terms, 3),
                }
            )
        hits.sort(key=lambda item: item["score"], reverse=True)
        return hits[: max(1, int(limit or 20))]

    def ls(self, namespace: str = "", limit: int = 100) -> List[Dict[str, Any]]:
        try:
            root = self._resolve(sanitize_path_part(namespace)) if namespace else self._resolve("")
        except ContextPathOutsideRootError:
            return []
        if not root.exists():
            return []
        items: List[Dict[str, Any]] = []
        for discovered_path in sorted(root.rglob("*")):
            try:
                path = self._resolve(str(discovered_path))
            except ContextPathOutsideRootError:
                continue
            if not path.is_file():
                continue
            items.append(
                {
                    "path": str(path),
                    "relativePath": str(path.relative_to(self.root)) if self._is_under_root(path) else str(path),
                    "merchantUri": merchant_uri_for_artifact(str(path.relative_to(self.root)) if self._is_under_root(path) else str(path)),
                    "bytes": path.stat().st_size,
                }
            )
            if len(items) >= max(1, int(limit or 100)):
                break
        return items

    def _resolve(self, path: str) -> Path:
        return resolve_context_path(self.root, path)

    def _is_under_root(self, path: Path) -> bool:
        return context_path_is_within_root(self.root, path)

    def _path_error(self, path: str) -> Dict[str, Any]:
        return {"success": False, "error": self.PATH_OUTSIDE_ROOT, "path": str(path or "")}


def offload_rows_if_needed(
    store: WorkspaceArtifactStore,
    namespace: str,
    name: str,
    rows: List[Dict[str, Any]],
    preview_rows: int,
) -> Dict[str, Any]:
    preview = rows[: max(0, preview_rows)]
    artifact = store.write_json(namespace, name, rows, preview_chars=0) if len(rows) > len(preview) else {}
    return {
        "rows": preview,
        "artifact": artifact,
        "offloaded": bool(artifact),
        "originalRowCount": len(rows),
    }


def sanitize_path_part(value: str) -> str:
    text = str(value or "misc").strip().replace("\\", "_").replace("/", "_")
    text = re.sub(r"[^A-Za-z0-9_.\-\u4e00-\u9fff]+", "_", text)
    return text or "misc"


def sanitize_file_name(value: str) -> str:
    text = sanitize_path_part(value)
    return text or "artifact.txt"


def artifact_snippets(content: str, terms: Iterable[str], limit: int) -> List[str]:
    text = str(content or "")
    lower = text.lower()
    snippets: List[str] = []
    for term in terms:
        pos = lower.find(term)
        if pos < 0:
            continue
        start = max(0, pos - 100)
        end = min(len(text), pos + len(term) + 160)
        snippet = text[start:end].replace("\n", " ").strip()
        if snippet and snippet not in snippets:
            snippets.append(snippet)
        if len(snippets) >= limit:
            break
    return snippets
