from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from merchant_ai.services.repositories import DatabaseStreamError


def build_grounded_canonical_json_encoder() -> json.JSONEncoder:
    """Return the one canonical encoder used for result bytes and seals."""

    return json.JSONEncoder(
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=str,
    )


def grounded_canonical_json_sha256(value: Any) -> str:
    """Hash the exact compact canonical JSON byte sequence for ``value``."""

    digest = hashlib.sha256()
    for chunk in build_grounded_canonical_json_encoder().iterencode(value):
        digest.update(chunk.encode("utf-8"))
    return digest.hexdigest()


class GroundedResultStreamCode(str, Enum):
    ROOT_INVALID = "RESULT_STREAM_ROOT_INVALID"
    PATH_INVALID = "RESULT_STREAM_PATH_INVALID"
    ARTIFACT_CONFLICT = "RESULT_STREAM_ARTIFACT_CONFLICT"
    ARTIFACT_BINDING_CHANGED = "RESULT_STREAM_ARTIFACT_BINDING_CHANGED"
    PUBLICATION_CONFLICT = "RESULT_STREAM_PUBLICATION_CONFLICT"
    ROW_INVALID = "RESULT_STREAM_ROW_INVALID"
    ROW_QUOTA_EXCEEDED = "RESULT_STREAM_ROW_QUOTA_EXCEEDED"
    BYTE_QUOTA_EXCEEDED = "RESULT_STREAM_BYTE_QUOTA_EXCEEDED"
    CANCELLED = "RESULT_STREAM_CANCELLED"
    TIMEOUT = "RESULT_STREAM_TIMEOUT"
    SOURCE_FAILED = "RESULT_STREAM_SOURCE_FAILED"
    WRITE_FAILED = "RESULT_STREAM_WRITE_FAILED"


class GroundedResultCoverage(str, Enum):
    ALL_ROWS = "ALL_ROWS"
    INACTIVE_PARTIAL = "INACTIVE_PARTIAL"


@dataclass(frozen=True)
class GroundedResultStreamLimits:
    preview_rows: int
    fetch_batch_rows: int
    max_rows: int
    max_bytes: int

    def __post_init__(self) -> None:
        if self.preview_rows < 0:
            raise ValueError("RESULT_STREAM_PREVIEW_LIMIT_INVALID")
        if self.fetch_batch_rows <= 0:
            raise ValueError("RESULT_STREAM_FETCH_BATCH_INVALID")
        if self.max_rows <= 0:
            raise ValueError("RESULT_STREAM_ROW_QUOTA_INVALID")
        if self.max_bytes < 2:
            raise ValueError("RESULT_STREAM_BYTE_QUOTA_INVALID")


@dataclass(frozen=True)
class GroundedResultPartialState:
    row_count: int
    byte_count: int
    preview_row_count: int
    coverage: str = GroundedResultCoverage.INACTIVE_PARTIAL.value
    complete: bool = False
    active: bool = False


class GroundedResultStreamingError(RuntimeError):
    def __init__(
        self,
        code: GroundedResultStreamCode,
        partial: GroundedResultPartialState,
        *,
        cause_type: str = "",
    ) -> None:
        self.code = code
        self.partial = partial
        self.cause_type = str(cause_type or "")
        message = code.value
        if self.cause_type:
            message = "%s:%s" % (message, self.cause_type)
        super().__init__(message)


@dataclass(frozen=True)
class GroundedResultArtifactReceipt:
    artifact_id: str
    rows_relative_path: str
    marker_relative_path: str
    # Hash of the exact compact canonical JSON bytes stored at rows_relative_path.
    rows_canonical_sha256: str
    content_address: str
    byte_count: int
    exact_row_count: int
    preview_rows: tuple[dict[str, Any], ...]
    preview_is_truncated: bool
    coverage: str = GroundedResultCoverage.ALL_ROWS.value
    complete: bool = True
    active: bool = True
    immutable: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifactId": self.artifact_id,
            "rowsRelativePath": self.rows_relative_path,
            "markerRelativePath": self.marker_relative_path,
            "rowsCanonicalSha256": self.rows_canonical_sha256,
            "contentAddress": self.content_address,
            "bytes": self.byte_count,
            "exactResultRowCount": self.exact_row_count,
            "previewRows": [dict(row) for row in self.preview_rows],
            "previewIsTruncated": self.preview_is_truncated,
            "resultCoverage": self.coverage,
            "complete": self.complete,
            "active": self.active,
            "immutable": self.immutable,
        }


class GroundedStreamingQuerySource(Protocol):
    def stream_query_batches(
        self,
        sql: str,
        params: Iterable[Any] | None = None,
        *,
        batch_size: int,
        cancel_events: Iterable[Any] | None = None,
        timeout_seconds: int | None = None,
        data_snapshot_contract: Any = None,
    ) -> Iterator[list[dict[str, Any]]]: ...


class GroundedResultStreamMaterializer:
    """Stream a complete result into an inactive-until-sealed JSON artifact.

    The caller supplies an already-created trusted root. Each materialization
    owns a new child namespace. The namespace is opened by descriptor and is
    revalidated before publication, so a path swap cannot redirect authority.
    A rows file without its matching immutable marker is never an active
    artifact. Only natural source EOF permits an ``ALL_ROWS`` receipt.
    """

    def __init__(self, artifact_root: Path | str) -> None:
        self.artifact_root = Path(os.path.abspath(str(artifact_root)))

    def materialize_query(
        self,
        source: GroundedStreamingQuerySource,
        *,
        sql: str,
        artifact_id: str,
        limits: GroundedResultStreamLimits,
        params: Iterable[Any] | None = None,
        cancel_events: Iterable[Any] | None = None,
        timeout_seconds: int | None = None,
        data_snapshot_contract: Any = None,
        rows_file_name: str = "rows.json",
    ) -> GroundedResultArtifactReceipt:
        try:
            batches = source.stream_query_batches(
                sql,
                params,
                batch_size=limits.fetch_batch_rows,
                cancel_events=cancel_events,
                timeout_seconds=timeout_seconds,
                data_snapshot_contract=data_snapshot_contract,
            )
        except DatabaseStreamError as exc:
            raise self._database_failure(exc, 0, 0, 0) from exc
        except Exception as exc:
            raise self._failure(
                GroundedResultStreamCode.SOURCE_FAILED,
                0,
                0,
                0,
                cause=exc,
            ) from exc
        return self.materialize_batches(
            batches,
            artifact_id=artifact_id,
            limits=limits,
            cancel_events=cancel_events,
            rows_file_name=rows_file_name,
        )

    def materialize_batches(
        self,
        batches: Iterable[Iterable[Mapping[str, Any]]],
        *,
        artifact_id: str,
        limits: GroundedResultStreamLimits,
        cancel_events: Iterable[Any] | None = None,
        rows_file_name: str = "rows.json",
    ) -> GroundedResultArtifactReceipt:
        artifact_component = _path_component(artifact_id)
        rows_component = _path_component(rows_file_name)
        if artifact_component.startswith(".") or rows_component.startswith("."):
            raise self._failure(GroundedResultStreamCode.PATH_INVALID, 0, 0, 0)

        cancellation_events = tuple(event for event in (cancel_events or ()) if event is not None)
        root_descriptor = -1
        artifact_descriptor = -1
        rows_descriptor = -1
        marker_descriptor = -1
        namespace_created = False
        committed = False
        temporary_rows_name = ".stream-write-%s.tmp" % uuid.uuid4().hex
        temporary_marker_name = ".stream-marker-%s.tmp" % uuid.uuid4().hex
        marker_name = _immutable_marker_name(rows_component)
        row_count = 0
        byte_count = 0
        preview_rows: list[dict[str, Any]] = []
        rows_identity: tuple[int, int] | None = None
        marker_identity: tuple[int, int] | None = None
        root_identity: tuple[int, int] | None = None
        artifact_identity: tuple[int, int] | None = None
        source_iterator: Iterator[Iterable[Mapping[str, Any]]] | None = None
        digest = hashlib.sha256()
        encoder = build_grounded_canonical_json_encoder()

        try:
            try:
                root_descriptor = os.open(
                    self.artifact_root,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                root_stat = os.fstat(root_descriptor)
                if not stat.S_ISDIR(root_stat.st_mode):
                    raise OSError(errno.ENOTDIR, "artifact root is not a directory")
                root_identity = (root_stat.st_dev, root_stat.st_ino)
            except OSError as exc:
                raise self._failure(
                    GroundedResultStreamCode.ROOT_INVALID,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc

            try:
                os.mkdir(artifact_component, 0o700, dir_fd=root_descriptor)
                namespace_created = True
                artifact_descriptor = os.open(
                    artifact_component,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=root_descriptor,
                )
                artifact_stat = os.fstat(artifact_descriptor)
                if not stat.S_ISDIR(artifact_stat.st_mode):
                    raise OSError(errno.ENOTDIR, "artifact namespace is not a directory")
                artifact_identity = (artifact_stat.st_dev, artifact_stat.st_ino)
            except FileExistsError as exc:
                raise self._failure(
                    GroundedResultStreamCode.ARTIFACT_CONFLICT,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise self._failure(
                    GroundedResultStreamCode.WRITE_FAILED,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc

            try:
                rows_descriptor = os.open(
                    temporary_rows_name,
                    os.O_WRONLY
                    | os.O_CREAT
                    | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=artifact_descriptor,
                )
            except OSError as exc:
                raise self._failure(
                    GroundedResultStreamCode.WRITE_FAILED,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc

            byte_count = self._emit(
                rows_descriptor,
                b"[",
                digest,
                byte_count,
                limits.max_bytes,
                row_count,
                len(preview_rows),
            )
            try:
                source_iterator = iter(batches)
            except Exception as exc:
                raise self._failure(
                    GroundedResultStreamCode.SOURCE_FAILED,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc
            while True:
                self._check_cancelled(
                    cancellation_events,
                    row_count,
                    byte_count,
                    len(preview_rows),
                )
                try:
                    batch = next(source_iterator)
                except StopIteration:
                    break
                except DatabaseStreamError as exc:
                    raise self._database_failure(
                        exc,
                        row_count,
                        byte_count,
                        len(preview_rows),
                    ) from exc
                except Exception as exc:
                    raise self._failure(
                        GroundedResultStreamCode.SOURCE_FAILED,
                        row_count,
                        byte_count,
                        len(preview_rows),
                        cause=exc,
                    ) from exc

                if batch is None or isinstance(batch, (str, bytes, bytearray, Mapping)):
                    raise self._failure(
                        GroundedResultStreamCode.ROW_INVALID,
                        row_count,
                        byte_count,
                        len(preview_rows),
                    )
                try:
                    batch_iterator = iter(batch)
                except TypeError as exc:
                    raise self._failure(
                        GroundedResultStreamCode.ROW_INVALID,
                        row_count,
                        byte_count,
                        len(preview_rows),
                        cause=exc,
                    ) from exc

                while True:
                    self._check_cancelled(
                        cancellation_events,
                        row_count,
                        byte_count,
                        len(preview_rows),
                    )
                    try:
                        raw_row = next(batch_iterator)
                    except StopIteration:
                        break
                    except Exception as exc:
                        raise self._failure(
                            GroundedResultStreamCode.SOURCE_FAILED,
                            row_count,
                            byte_count,
                            len(preview_rows),
                            cause=exc,
                        ) from exc
                    if row_count >= limits.max_rows:
                        raise self._failure(
                            GroundedResultStreamCode.ROW_QUOTA_EXCEEDED,
                            row_count,
                            byte_count,
                            len(preview_rows),
                        )
                    if not isinstance(raw_row, Mapping):
                        raise self._failure(
                            GroundedResultStreamCode.ROW_INVALID,
                            row_count,
                            byte_count,
                            len(preview_rows),
                        )
                    try:
                        row = dict(raw_row)
                    except Exception as exc:
                        raise self._failure(
                            GroundedResultStreamCode.ROW_INVALID,
                            row_count,
                            byte_count,
                            len(preview_rows),
                            cause=exc,
                        ) from exc
                    if any(not isinstance(key, str) for key in row):
                        raise self._failure(
                            GroundedResultStreamCode.ROW_INVALID,
                            row_count,
                            byte_count,
                            len(preview_rows),
                        )
                    if row_count:
                        byte_count = self._emit(
                            rows_descriptor,
                            b",",
                            digest,
                            byte_count,
                            limits.max_bytes,
                            row_count,
                            len(preview_rows),
                        )
                    capture_preview = len(preview_rows) < limits.preview_rows
                    preview_chunks: list[str] = []
                    try:
                        for chunk in encoder.iterencode(row):
                            if capture_preview:
                                preview_chunks.append(chunk)
                            byte_count = self._emit(
                                rows_descriptor,
                                chunk.encode("utf-8"),
                                digest,
                                byte_count,
                                limits.max_bytes,
                                row_count,
                                len(preview_rows),
                            )
                    except (TypeError, ValueError, UnicodeError) as exc:
                        raise self._failure(
                            GroundedResultStreamCode.ROW_INVALID,
                            row_count,
                            byte_count,
                            len(preview_rows),
                            cause=exc,
                        ) from exc
                    if capture_preview:
                        try:
                            preview_value = json.loads("".join(preview_chunks))
                        except (json.JSONDecodeError, UnicodeError) as exc:
                            raise self._failure(
                                GroundedResultStreamCode.ROW_INVALID,
                                row_count,
                                byte_count,
                                len(preview_rows),
                                cause=exc,
                            ) from exc
                        if not isinstance(preview_value, dict):
                            raise self._failure(
                                GroundedResultStreamCode.ROW_INVALID,
                                row_count,
                                byte_count,
                                len(preview_rows),
                            )
                        preview_rows.append(preview_value)
                    row_count += 1

            self._check_cancelled(
                cancellation_events,
                row_count,
                byte_count,
                len(preview_rows),
            )
            byte_count = self._emit(
                rows_descriptor,
                b"]",
                digest,
                byte_count,
                limits.max_bytes,
                row_count,
                len(preview_rows),
            )
            os.fsync(rows_descriptor)
            os.fchmod(rows_descriptor, 0o400)
            rows_stat = os.fstat(rows_descriptor)
            rows_identity = (rows_stat.st_dev, rows_stat.st_ino)
            rows_sha256 = digest.hexdigest()

            marker_payload = ("%s\n" % rows_sha256).encode("ascii")
            marker_descriptor = os.open(
                temporary_marker_name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=artifact_descriptor,
            )
            _write_all(marker_descriptor, marker_payload)
            os.fsync(marker_descriptor)
            os.fchmod(marker_descriptor, 0o400)
            marker_stat = os.fstat(marker_descriptor)
            marker_identity = (marker_stat.st_dev, marker_stat.st_ino)

            self._validate_binding(
                root_descriptor,
                artifact_component,
                root_identity,
                artifact_identity,
                row_count,
                byte_count,
                len(preview_rows),
            )
            self._check_cancelled(
                cancellation_events,
                row_count,
                byte_count,
                len(preview_rows),
            )
            try:
                os.link(
                    temporary_rows_name,
                    rows_component,
                    src_dir_fd=artifact_descriptor,
                    dst_dir_fd=artifact_descriptor,
                    follow_symlinks=False,
                )
                os.link(
                    temporary_marker_name,
                    marker_name,
                    src_dir_fd=artifact_descriptor,
                    dst_dir_fd=artifact_descriptor,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise self._failure(
                    GroundedResultStreamCode.PUBLICATION_CONFLICT,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc
            except OSError as exc:
                raise self._failure(
                    GroundedResultStreamCode.WRITE_FAILED,
                    row_count,
                    byte_count,
                    len(preview_rows),
                    cause=exc,
                ) from exc
            self._validate_owned_entry(
                artifact_descriptor,
                rows_component,
                rows_identity,
                row_count,
                byte_count,
                len(preview_rows),
            )
            self._validate_owned_entry(
                artifact_descriptor,
                marker_name,
                marker_identity,
                row_count,
                byte_count,
                len(preview_rows),
            )
            os.unlink(temporary_rows_name, dir_fd=artifact_descriptor)
            os.unlink(temporary_marker_name, dir_fd=artifact_descriptor)
            os.fsync(artifact_descriptor)
            self._validate_binding(
                root_descriptor,
                artifact_component,
                root_identity,
                artifact_identity,
                row_count,
                byte_count,
                len(preview_rows),
            )
            self._validate_published_content(
                artifact_descriptor,
                rows_component,
                marker_name,
                rows_identity,
                marker_identity,
                rows_sha256,
                byte_count,
                marker_payload,
                row_count,
                byte_count,
                len(preview_rows),
            )
            committed = True
            return GroundedResultArtifactReceipt(
                artifact_id=artifact_component,
                rows_relative_path="%s/%s" % (artifact_component, rows_component),
                marker_relative_path="%s/%s" % (artifact_component, marker_name),
                rows_canonical_sha256=rows_sha256,
                content_address="sha256:%s" % rows_sha256,
                byte_count=byte_count,
                exact_row_count=row_count,
                preview_rows=tuple(preview_rows),
                preview_is_truncated=row_count > len(preview_rows),
            )
        except GroundedResultStreamingError:
            raise
        except OSError as exc:
            raise self._failure(
                GroundedResultStreamCode.WRITE_FAILED,
                row_count,
                byte_count,
                len(preview_rows),
                cause=exc,
            ) from exc
        finally:
            if source_iterator is not None:
                close = getattr(source_iterator, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass
            if not committed and artifact_descriptor >= 0:
                _unlink_owned(
                    artifact_descriptor,
                    marker_name,
                    marker_identity,
                )
                _unlink_owned(
                    artifact_descriptor,
                    rows_component,
                    rows_identity,
                )
            if artifact_descriptor >= 0:
                _unlink_name(artifact_descriptor, temporary_marker_name)
                _unlink_name(artifact_descriptor, temporary_rows_name)
            for descriptor in (marker_descriptor, rows_descriptor, artifact_descriptor):
                if descriptor >= 0:
                    try:
                        os.close(descriptor)
                    except OSError:
                        pass
            if namespace_created and not committed and root_descriptor >= 0:
                try:
                    current = os.stat(
                        artifact_component,
                        dir_fd=root_descriptor,
                        follow_symlinks=False,
                    )
                    if artifact_identity == (current.st_dev, current.st_ino):
                        os.rmdir(artifact_component, dir_fd=root_descriptor)
                        os.fsync(root_descriptor)
                except OSError:
                    pass
            if root_descriptor >= 0:
                try:
                    os.close(root_descriptor)
                except OSError:
                    pass

    def _validate_binding(
        self,
        root_descriptor: int,
        artifact_component: str,
        root_identity: tuple[int, int] | None,
        artifact_identity: tuple[int, int] | None,
        row_count: int,
        byte_count: int,
        preview_count: int,
    ) -> None:
        try:
            root_stat = os.stat(self.artifact_root, follow_symlinks=False)
            artifact_stat = os.stat(
                artifact_component,
                dir_fd=root_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise self._failure(
                GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                row_count,
                byte_count,
                preview_count,
                cause=exc,
            ) from exc
        if (
            root_identity != (root_stat.st_dev, root_stat.st_ino)
            or artifact_identity != (artifact_stat.st_dev, artifact_stat.st_ino)
            or not stat.S_ISDIR(root_stat.st_mode)
            or not stat.S_ISDIR(artifact_stat.st_mode)
        ):
            raise self._failure(
                GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                row_count,
                byte_count,
                preview_count,
            )

    @staticmethod
    def _validate_owned_entry(
        directory_descriptor: int,
        name: str,
        identity: tuple[int, int] | None,
        row_count: int,
        byte_count: int,
        preview_count: int,
    ) -> None:
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            observed = os.fstat(descriptor)
            if (
                identity != (observed.st_dev, observed.st_ino)
                or not stat.S_ISREG(observed.st_mode)
            ):
                raise GroundedResultStreamMaterializer._failure(
                    GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                    row_count,
                    byte_count,
                    preview_count,
                )
        except GroundedResultStreamingError:
            raise
        except OSError as exc:
            raise GroundedResultStreamMaterializer._failure(
                GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                row_count,
                byte_count,
                preview_count,
                cause=exc,
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _validate_published_content(
        directory_descriptor: int,
        rows_name: str,
        marker_name: str,
        rows_identity: tuple[int, int] | None,
        marker_identity: tuple[int, int] | None,
        expected_sha256: str,
        expected_bytes: int,
        expected_marker: bytes,
        row_count: int,
        byte_count: int,
        preview_count: int,
    ) -> None:
        rows_descriptor = -1
        marker_descriptor = -1
        try:
            rows_descriptor = os.open(
                rows_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            rows_stat = os.fstat(rows_descriptor)
            if (
                rows_identity != (rows_stat.st_dev, rows_stat.st_ino)
                or not stat.S_ISREG(rows_stat.st_mode)
                or rows_stat.st_size != expected_bytes
            ):
                raise GroundedResultStreamMaterializer._failure(
                    GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                    row_count,
                    byte_count,
                    preview_count,
                )
            observed_digest = hashlib.sha256()
            while True:
                chunk = os.read(rows_descriptor, 1024 * 1024)
                if not chunk:
                    break
                observed_digest.update(chunk)
            if observed_digest.hexdigest() != expected_sha256:
                raise GroundedResultStreamMaterializer._failure(
                    GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                    row_count,
                    byte_count,
                    preview_count,
                )

            marker_descriptor = os.open(
                marker_name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_descriptor,
            )
            marker_stat = os.fstat(marker_descriptor)
            if (
                marker_identity != (marker_stat.st_dev, marker_stat.st_ino)
                or not stat.S_ISREG(marker_stat.st_mode)
                or marker_stat.st_size != len(expected_marker)
            ):
                raise GroundedResultStreamMaterializer._failure(
                    GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                    row_count,
                    byte_count,
                    preview_count,
                )
            observed_marker = bytearray()
            while True:
                chunk = os.read(marker_descriptor, 1024)
                if not chunk:
                    break
                observed_marker.extend(chunk)
                if len(observed_marker) > len(expected_marker):
                    break
            if bytes(observed_marker) != expected_marker:
                raise GroundedResultStreamMaterializer._failure(
                    GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                    row_count,
                    byte_count,
                    preview_count,
                )
        except GroundedResultStreamingError:
            raise
        except OSError as exc:
            raise GroundedResultStreamMaterializer._failure(
                GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
                row_count,
                byte_count,
                preview_count,
                cause=exc,
            ) from exc
        finally:
            if marker_descriptor >= 0:
                os.close(marker_descriptor)
            if rows_descriptor >= 0:
                os.close(rows_descriptor)

    @staticmethod
    def _emit(
        descriptor: int,
        content: bytes,
        digest: Any,
        byte_count: int,
        max_bytes: int,
        row_count: int,
        preview_count: int,
    ) -> int:
        projected = byte_count + len(content)
        if projected > max_bytes:
            raise GroundedResultStreamMaterializer._failure(
                GroundedResultStreamCode.BYTE_QUOTA_EXCEEDED,
                row_count,
                byte_count,
                preview_count,
            )
        try:
            _write_all(descriptor, content)
        except OSError as exc:
            raise GroundedResultStreamMaterializer._failure(
                GroundedResultStreamCode.WRITE_FAILED,
                row_count,
                byte_count,
                preview_count,
                cause=exc,
            ) from exc
        digest.update(content)
        return projected

    @staticmethod
    def _check_cancelled(
        cancel_events: Sequence[Any],
        row_count: int,
        byte_count: int,
        preview_count: int,
    ) -> None:
        if any(bool(getattr(event, "is_set", lambda: False)()) for event in cancel_events):
            raise GroundedResultStreamMaterializer._failure(
                GroundedResultStreamCode.CANCELLED,
                row_count,
                byte_count,
                preview_count,
            )

    @staticmethod
    def _database_failure(
        exc: DatabaseStreamError,
        row_count: int,
        byte_count: int,
        preview_count: int,
    ) -> GroundedResultStreamingError:
        code = GroundedResultStreamCode.SOURCE_FAILED
        if exc.code == DatabaseStreamError.CANCELLED:
            code = GroundedResultStreamCode.CANCELLED
        elif exc.code == DatabaseStreamError.TIMEOUT:
            code = GroundedResultStreamCode.TIMEOUT
        return GroundedResultStreamMaterializer._failure(
            code,
            row_count,
            byte_count,
            preview_count,
            cause=exc,
        )

    @staticmethod
    def _failure(
        code: GroundedResultStreamCode,
        row_count: int,
        byte_count: int,
        preview_count: int,
        *,
        cause: BaseException | None = None,
    ) -> GroundedResultStreamingError:
        return GroundedResultStreamingError(
            code,
            GroundedResultPartialState(
                row_count=max(0, int(row_count or 0)),
                byte_count=max(0, int(byte_count or 0)),
                preview_row_count=max(0, int(preview_count or 0)),
            ),
            cause_type=type(cause).__name__ if cause is not None else "",
        )


def _path_component(value: str) -> str:
    component = str(value or "")
    if (
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        or "\x00" in component
    ):
        raise GroundedResultStreamMaterializer._failure(
            GroundedResultStreamCode.PATH_INVALID,
            0,
            0,
            0,
        )
    return component


def _immutable_marker_name(file_name: str) -> str:
    identity = hashlib.sha256(file_name.encode("utf-8")).hexdigest()
    return ".artifact-immutable-%s.sha256" % identity


def _write_all(descriptor: int, content: bytes) -> None:
    remaining = memoryview(content)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError(errno.EIO, "short result artifact write")
        remaining = remaining[written:]


def _unlink_name(directory_descriptor: int, name: str) -> None:
    try:
        os.unlink(name, dir_fd=directory_descriptor)
    except OSError:
        pass


def _unlink_owned(
    directory_descriptor: int,
    name: str,
    identity: tuple[int, int] | None,
) -> None:
    if identity is None:
        return
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_descriptor,
        )
        observed = os.fstat(descriptor)
        if identity == (observed.st_dev, observed.st_ino) and stat.S_ISREG(observed.st_mode):
            os.unlink(name, dir_fd=directory_descriptor)
    except OSError:
        pass
    finally:
        if descriptor >= 0:
            os.close(descriptor)
