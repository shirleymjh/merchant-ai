from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import fcntl

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
    PATH_RESERVED = "ARTIFACT_PATH_RESERVED"
    IMMUTABLE_CONFLICT = "ARTIFACT_IMMUTABLE_CONFLICT"
    IMMUTABLE_STATE_INVALID = "ARTIFACT_IMMUTABLE_STATE_INVALID"

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

    def write_json(
        self,
        namespace: str,
        name: str,
        payload: Any,
        preview_chars: int | None = None,
        *,
        immutable: bool = False,
    ) -> Dict[str, Any]:
        text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        return self.write_text(
            namespace,
            name if name.endswith(".json") else "%s.json" % name,
            text,
            preview_chars=preview_chars,
            immutable=immutable,
        )

    def write_text(
        self,
        namespace: str,
        name: str,
        content: str,
        preview_chars: int | None = None,
        *,
        immutable: bool = False,
    ) -> Dict[str, Any]:
        relative_path = Path(sanitize_path_part(namespace or "misc")) / sanitize_file_name(name or "artifact.txt")
        if _is_internal_artifact_path(relative_path):
            return {"success": False, "error": self.PATH_RESERVED, "path": str(relative_path)}
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
        encoded = text.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        artifact_is_immutable = False
        idempotent = False
        try:
            with _artifact_target_lock(target):
                # The lock protects store writers; resolving again also rejects
                # a parent or target symlink swapped in after directory creation.
                target = self._resolve(str(relative_path))
                marker_path = _immutable_marker_path(target)
                immutable_digest = _read_immutable_digest(marker_path)
                if immutable_digest is not None:
                    recovered_missing_target = False
                    if not target.exists():
                        if immutable and digest == immutable_digest:
                            _atomic_create_text(target, encoded)
                            recovered_missing_target = True
                        else:
                            raise _ImmutableStateInvalidError
                    existing_digest, _ = _file_digest(target)
                    if existing_digest != immutable_digest:
                        raise _ImmutableStateInvalidError
                    if digest != immutable_digest:
                        raise _ImmutableConflictError
                    artifact_is_immutable = True
                    idempotent = not recovered_missing_target
                elif immutable:
                    if target.exists():
                        existing_digest, _ = _file_digest(target)
                        if existing_digest != digest:
                            raise _ImmutableConflictError
                        idempotent = True
                        _atomic_write_text(marker_path, "%s\n" % digest)
                    else:
                        # Commit the immutable intent before publishing data.
                        # Readers that require immutable evidence ignore the
                        # path until both files exist and their hashes agree.
                        _atomic_write_text(marker_path, "%s\n" % digest)
                        try:
                            _atomic_create_text(target, encoded)
                        except FileExistsError:
                            # A writer outside this store may have won the path.
                            # Only byte-identical content can be adopted.
                            target = self._resolve(str(relative_path))
                            existing_digest, _ = _file_digest(target)
                            if existing_digest != digest:
                                raise _ImmutableConflictError
                            idempotent = True
                    artifact_is_immutable = True
                else:
                    _atomic_write_text(target, text)
        except _ImmutableConflictError:
            return {
                "success": False,
                "error": self.IMMUTABLE_CONFLICT,
                "path": str(relative_path),
                "sha256": digest,
            }
        except _ImmutableStateInvalidError:
            return {
                "success": False,
                "error": self.IMMUTABLE_STATE_INVALID,
                "path": str(relative_path),
                "sha256": digest,
            }
        except ContextPathOutsideRootError:
            return self._path_error(str(relative_path))
        except OSError:
            return {"success": False, "error": "ARTIFACT_WRITE_FAILED", "path": str(relative_path)}
        preview_limit = max(0, int(preview_chars if preview_chars is not None else self.settings.context_file_inline_max_chars))
        return {
            "success": True,
            "path": str(target),
            "relativePath": str(target.relative_to(self.root)),
            "merchantUri": merchant_uri_for_artifact(str(target.relative_to(self.root)), namespace=namespace or "misc"),
            "bytes": len(encoded),
            "estimatedChars": len(text),
            "sha256": digest,
            "contentAddress": "sha256:%s" % digest,
            "immutable": artifact_is_immutable,
            "idempotent": idempotent,
            "preview": text[:preview_limit],
            "truncated": len(text) > preview_limit,
        }

    def read(
        self,
        path: str,
        offset: int = 0,
        max_chars: int | None = None,
        *,
        require_immutable: bool = False,
    ) -> Dict[str, Any]:
        try:
            target = self._resolve(path)
        except ContextPathOutsideRootError:
            return self._path_error(path)
        if _is_internal_artifact_path(target):
            return {"success": False, "error": "ARTIFACT_NOT_FOUND", "path": path}
        if not target.exists() or not target.is_file():
            return {"success": False, "error": "ARTIFACT_NOT_FOUND", "path": path}
        try:
            with _artifact_target_lock(target):
                target = self._resolve(path)
                marker_path = _immutable_marker_path(target)
                immutable_digest = _read_immutable_digest(marker_path)
                if require_immutable and immutable_digest is None:
                    raise _ImmutableStateInvalidError
                encoded = target.read_bytes()
                actual_digest = hashlib.sha256(encoded).hexdigest()
                if (
                    immutable_digest is not None
                    and actual_digest != immutable_digest
                ):
                    raise _ImmutableStateInvalidError
                text = encoded.decode("utf-8")
        except _ImmutableStateInvalidError:
            return {
                "success": False,
                "error": self.IMMUTABLE_STATE_INVALID,
                "path": path,
            }
        except (OSError, UnicodeError):
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
            "sha256": actual_digest,
            "contentAddress": "sha256:%s" % actual_digest,
            "immutable": immutable_digest is not None,
        }

    def grep(
        self,
        query: str,
        limit: int = 20,
        *,
        require_immutable: bool = False,
    ) -> List[Dict[str, Any]]:
        terms = _artifact_search_terms(query)
        if not terms:
            return []
        hits: List[Dict[str, Any]] = []
        for discovered_path in sorted(self.root.rglob("*")):
            if _is_internal_artifact_path(discovered_path):
                continue
            try:
                path = self._resolve(str(discovered_path))
            except ContextPathOutsideRootError:
                continue
            if not path.is_file():
                continue
            relative_path = str(path.relative_to(self.root))
            read_result = self.read(
                relative_path,
                max_chars=max(1, int(path.stat().st_size) + 1),
                require_immutable=require_immutable,
            )
            if not read_result.get("success"):
                continue
            text = str(read_result.get("content") or "")
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

    def ls(
        self,
        namespace: str = "",
        limit: int = 100,
        *,
        require_immutable: bool = False,
    ) -> List[Dict[str, Any]]:
        try:
            root = self._resolve(sanitize_path_part(namespace)) if namespace else self._resolve("")
        except ContextPathOutsideRootError:
            return []
        if not root.exists():
            return []
        items: List[Dict[str, Any]] = []
        for discovered_path in sorted(root.rglob("*")):
            if _is_internal_artifact_path(discovered_path):
                continue
            try:
                path = self._resolve(str(discovered_path))
            except ContextPathOutsideRootError:
                continue
            if not path.is_file():
                continue
            try:
                with _artifact_target_lock(path):
                    path = self._resolve(str(path.relative_to(self.root)))
                    immutable_digest = _read_immutable_digest(
                        _immutable_marker_path(path)
                    )
                    if require_immutable and immutable_digest is None:
                        raise _ImmutableStateInvalidError
                    actual_digest, byte_count = _file_digest(path)
                    if (
                        immutable_digest is not None
                        and actual_digest != immutable_digest
                    ):
                        raise _ImmutableStateInvalidError
            except (
                ContextPathOutsideRootError,
                OSError,
                _ImmutableStateInvalidError,
            ):
                continue
            items.append(
                {
                    "path": str(path),
                    "relativePath": str(path.relative_to(self.root)) if self._is_under_root(path) else str(path),
                    "merchantUri": merchant_uri_for_artifact(str(path.relative_to(self.root)) if self._is_under_root(path) else str(path)),
                    "bytes": byte_count,
                    "sha256": actual_digest,
                    "contentAddress": "sha256:%s" % actual_digest,
                    "immutable": immutable_digest is not None,
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
    text = _replace_disallowed_path_runs(text)
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


_INTERNAL_TEMPORARY_PREFIX = ".artifact-write-"
_INTERNAL_LOCK_PREFIX = ".artifact-lock-"
_INTERNAL_IMMUTABLE_PREFIX = ".artifact-immutable-"


class _ImmutableConflictError(RuntimeError):
    pass


class _ImmutableStateInvalidError(RuntimeError):
    pass


def _atomic_write_text(target: Path, text: str) -> None:
    """Durably replace one artifact without exposing a partial target file."""

    temporary_path = _write_synced_temporary(target.parent, text.encode("utf-8"))
    try:
        os.replace(temporary_path, target)
        temporary_path = None
        _fsync_directory(target.parent)
    finally:
        _remove_temporary_file(temporary_path)


def _atomic_create_text(target: Path, content: bytes) -> None:
    """Publish an immutable artifact only when its target does not exist."""

    temporary_path = _write_synced_temporary(target.parent, content)
    try:
        os.link(temporary_path, target)
        temporary_path.unlink()
        temporary_path = None
        _fsync_directory(target.parent)
    finally:
        _remove_temporary_file(temporary_path)


def _write_synced_temporary(directory: Path, content: bytes) -> Path:
    descriptor = -1
    temporary_path: Path | None = None
    completed = False
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(directory),
            prefix=_INTERNAL_TEMPORARY_PREFIX,
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary_file:
            descriptor = -1
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        completed = True
        return temporary_path
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not completed:
            _remove_temporary_file(temporary_path)


def _remove_temporary_file(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


@contextmanager
def _artifact_target_lock(target: Path) -> Iterator[None]:
    identity = hashlib.sha256(target.name.encode("utf-8")).hexdigest()
    lock_path = target.parent / ("%s%s.lock" % (_INTERNAL_LOCK_PREFIX, identity))
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(lock_path, flags, 0o600)
    locked = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        try:
            if locked:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _immutable_marker_path(target: Path) -> Path:
    identity = hashlib.sha256(target.name.encode("utf-8")).hexdigest()
    return target.parent / ("%s%s.sha256" % (_INTERNAL_IMMUTABLE_PREFIX, identity))


def _read_immutable_digest(marker_path: Path) -> str | None:
    if not marker_path.exists():
        return None
    if marker_path.is_symlink() or not marker_path.is_file():
        raise _ImmutableStateInvalidError
    try:
        digest = marker_path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise _ImmutableStateInvalidError from exc
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _ImmutableStateInvalidError
    return digest


def _file_digest(path: Path) -> tuple[str, int]:
    if path.is_symlink() or not path.is_file():
        raise _ImmutableStateInvalidError
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as artifact_file:
            while True:
                chunk = artifact_file.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                byte_count += len(chunk)
    except OSError as exc:
        raise _ImmutableStateInvalidError from exc
    return digest.hexdigest(), byte_count


def _is_internal_artifact_path(path: Path | str) -> bool:
    name = Path(path).name
    return name.startswith(
        (
            _INTERNAL_TEMPORARY_PREFIX,
            _INTERNAL_LOCK_PREFIX,
            _INTERNAL_IMMUTABLE_PREFIX,
        )
    )


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_disallowed_path_runs(value: str) -> str:
    output: List[str] = []
    replacing = False
    for character in value:
        if _is_artifact_path_character(character):
            output.append(character)
            replacing = False
            continue
        if not replacing:
            output.append("_")
        replacing = True
    return "".join(output)


def _is_artifact_path_character(character: str) -> bool:
    return (
        "A" <= character <= "Z"
        or "a" <= character <= "z"
        or "0" <= character <= "9"
        or character in "_.-"
        or "\u4e00" <= character <= "\u9fff"
    )


def _artifact_search_terms(value: str) -> List[str]:
    """Tokenize artifact search text with the filesystem's stable ASCII/CJK grammar."""

    text = str(value or "")
    terms: List[str] = []
    cursor = 0
    while cursor < len(text):
        character = text[cursor]
        if _is_ascii_letter(character) or character == "_":
            end = cursor + 1
            while end < len(text) and (
                _is_ascii_letter(text[end]) or _is_ascii_digit(text[end]) or text[end] == "_"
            ):
                end += 1
            terms.append(text[cursor:end].lower())
            cursor = end
            continue
        if "\u4e00" <= character <= "\u9fff":
            end = cursor + 1
            while end < len(text) and "\u4e00" <= text[end] <= "\u9fff":
                end += 1
            if end - cursor >= 2:
                terms.append(text[cursor:end].lower())
            cursor = end
            continue
        cursor += 1
    return terms


def _is_ascii_letter(character: str) -> bool:
    return "A" <= character <= "Z" or "a" <= character <= "z"


def _is_ascii_digit(character: str) -> bool:
    return "0" <= character <= "9"
