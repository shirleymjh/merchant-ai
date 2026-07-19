from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from merchant_ai.config import Settings


CONTEXT_WORKSPACE_SCHEMA_VERSION = 1


class GroundedContextWorkspaceError(RuntimeError):
    """Raised when a run-scoped context workspace cannot be trusted."""


def _directory_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _validate_component(component: str) -> str:
    value = str(component or "")
    if not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise GroundedContextWorkspaceError(
            "GROUNDED_CONTEXT_PATH_COMPONENT_INVALID"
        )
    return value


def _open_directory_beneath(
    root: Path,
    components: tuple[str, ...] = (),
    *,
    create: bool = False,
    exclusive_final: bool = False,
) -> int:
    """Open a descendant by dirfd without following any path symlink."""

    descriptor = os.open(root, _directory_open_flags())
    try:
        for index, raw_component in enumerate(components):
            component = _validate_component(raw_component)
            final = index == len(components) - 1
            if create:
                try:
                    os.mkdir(component, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    if exclusive_final and final:
                        raise GroundedContextWorkspaceError(
                            "GROUNDED_CONTEXT_DIRECTORY_ALREADY_EXISTS"
                        )
            child_descriptor = os.open(
                component,
                _directory_open_flags(),
                dir_fd=descriptor,
            )
            child_stat = os.fstat(child_descriptor)
            if not stat.S_ISDIR(child_stat.st_mode):
                os.close(child_descriptor)
                raise GroundedContextWorkspaceError(
                    "GROUNDED_CONTEXT_DIRECTORY_INVALID"
                )
            os.close(descriptor)
            descriptor = child_descriptor
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def _read_regular_file_at(directory_descriptor: int, name: str) -> bytes:
    component = _validate_component(name)
    descriptor = os.open(
        component,
        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
        dir_fd=directory_descriptor,
    )
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise GroundedContextWorkspaceError(
                "GROUNDED_CONTEXT_FILE_INVALID"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _atomic_write_at(
    directory_descriptor: int,
    name: str,
    encoded: bytes,
    *,
    error_code: str,
) -> None:
    component = _validate_component(name)
    temporary = ".%s.%s.tmp" % (component, uuid.uuid4().hex)
    descriptor = -1
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary,
            component,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except OSError as exc:
        raise GroundedContextWorkspaceError(error_code) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path_component(value: str, prefix: str) -> str:
    raw = str(value or "").strip()
    if raw and raw not in {".", ".."} and len(raw) <= 128 and all(
        character.isascii()
        and (character.isalnum() or character in {"_", "-", "."})
        for character in raw
    ):
        return raw
    return "%s_%s" % (prefix, hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24])


def grounded_context_owner_fingerprint(
    merchant_id: str,
    access_role: str,
    user_scope: Mapping[str, Any] | None,
) -> str:
    scope = dict(user_scope or {})
    identity = {
        "merchantId": str(merchant_id or "").strip(),
        "tenantId": str(scope.get("tenantId") or scope.get("tenant_id") or "").strip(),
        "principalId": str(
            scope.get("principalId")
            or scope.get("principal_id")
            or scope.get("userId")
            or scope.get("user_id")
            or ""
        ).strip(),
        "accessRole": str(access_role or scope.get("role") or "").strip(),
        "permissions": sorted(
            {
                str(item).strip()
                for item in scope.get("permissions") or []
                if str(item).strip()
            }
        ),
        "region": str(scope.get("region") or "").strip(),
        "storeIds": sorted(
            {
                str(item).strip()
                for item in scope.get("storeIds") or scope.get("store_ids") or []
                if str(item).strip()
            }
        ),
        "extendedServerScopeFingerprint": _stable_hash(scope),
    }
    return _stable_hash(identity)


def validated_grounded_query_artifact_roots(
    workspace_root: Path,
    artifact_root: Path | str,
) -> tuple[Path, Path]:
    """Open the pre-created publication and staging roots without symlinks."""

    trusted_workspace = Path(workspace_root).resolve(strict=True)
    requested_publication = Path(
        os.path.abspath(str(artifact_root or ""))
    )
    try:
        publication_components = tuple(
            requested_publication.relative_to(trusted_workspace).parts
        )
    except ValueError as exc:
        raise GroundedContextWorkspaceError(
            "QUERY_RESULT_ARTIFACT_ROOT_OUTSIDE_WORKSPACE"
        ) from exc
    if not publication_components:
        raise GroundedContextWorkspaceError(
            "QUERY_RESULT_ARTIFACT_ROOT_INVALID"
        )
    staging_components = (
        *publication_components[:-1],
        "staging",
        "query_results",
    )
    descriptors: list[int] = []
    try:
        descriptors.append(
            _open_directory_beneath(
                trusted_workspace,
                publication_components,
            )
        )
        descriptors.append(
            _open_directory_beneath(
                trusted_workspace,
                staging_components,
            )
        )
    except (OSError, GroundedContextWorkspaceError) as exc:
        raise GroundedContextWorkspaceError(
            "QUERY_RESULT_PRECREATED_ROOT_REQUIRED"
        ) from exc
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
    return (
        trusted_workspace.joinpath(*publication_components),
        trusted_workspace.joinpath(*staging_components),
    )


@dataclass(frozen=True)
class GroundedContextWorkspace:
    """One identity-bound filesystem boundary for a Grounded Core run."""

    root: Path
    artifacts_root: Path
    staging_root: Path
    core_scratch_root: Path
    subagents_root: Path
    thread_fingerprint: str
    run_fingerprint: str
    owner_fingerprint: str
    request_fingerprint: str

    @classmethod
    def open(
        cls,
        settings: Settings,
        *,
        thread_id: str,
        run_id: str,
        merchant_id: str,
        access_role: str,
        user_scope: Mapping[str, Any] | None,
        question: str,
    ) -> "GroundedContextWorkspace":
        workspace_root = settings.resolved_workspace_path.resolve()
        workspace_root.mkdir(parents=True, exist_ok=True)
        workspace_root = workspace_root.resolve(strict=True)
        safe_thread = _path_component(thread_id, "thread")
        safe_run = _path_component(run_id, "run")
        root_components = (
            "threads",
            safe_thread,
            "runs",
            safe_run,
            "outputs",
        )
        try:
            root_descriptor = _open_directory_beneath(
                workspace_root,
                root_components,
                create=True,
            )
            os.close(root_descriptor)
            root = workspace_root.joinpath(*root_components)
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedContextWorkspaceError(
                "GROUNDED_CONTEXT_WORKSPACE_OUTSIDE_ROOT"
            ) from exc

        owner_fingerprint = grounded_context_owner_fingerprint(
            merchant_id,
            access_role,
            user_scope,
        )
        thread_fingerprint = _stable_hash({"threadId": str(thread_id or "")})
        run_fingerprint = _stable_hash({"runId": str(run_id or "")})
        request_fingerprint = _stable_hash(
            {
                "threadFingerprint": thread_fingerprint,
                "runFingerprint": run_fingerprint,
                "ownerFingerprint": owner_fingerprint,
                "question": str(question or "").strip(),
            }
        )
        manifest = {
            "schemaVersion": CONTEXT_WORKSPACE_SCHEMA_VERSION,
            "workspaceKind": "GROUNDED_CONTEXT",
            "threadFingerprint": thread_fingerprint,
            "runFingerprint": run_fingerprint,
            "ownerFingerprint": owner_fingerprint,
            "requestFingerprint": request_fingerprint,
        }
        cls._create_or_validate_manifest(root, manifest)

        directory_components = (
            ("artifacts",),
            ("staging", "query_results"),
            ("scratch", "core"),
            ("scratch", "subagents"),
        )
        try:
            for components in directory_components:
                descriptor = _open_directory_beneath(
                    root,
                    components,
                    create=True,
                )
                os.close(descriptor)
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedContextWorkspaceError(
                "GROUNDED_CONTEXT_DIRECTORY_OUTSIDE_ROOT"
            ) from exc
        artifacts_root = root / "artifacts"
        staging_root = root / "staging" / "query_results"
        core_scratch_root = root / "scratch" / "core"
        subagents_root = root / "scratch" / "subagents"
        return cls(
            root=root,
            artifacts_root=artifacts_root,
            staging_root=staging_root,
            core_scratch_root=core_scratch_root,
            subagents_root=subagents_root,
            thread_fingerprint=thread_fingerprint,
            run_fingerprint=run_fingerprint,
            owner_fingerprint=owner_fingerprint,
            request_fingerprint=request_fingerprint,
        )

    @staticmethod
    def _create_or_validate_manifest(root: Path, expected: dict[str, Any]) -> None:
        root_descriptor = -1
        lock_descriptor = -1
        try:
            root_descriptor = _open_directory_beneath(root)
            lock_descriptor = os.open(
                ".context.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_descriptor,
            )
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise GroundedContextWorkspaceError(
                    "GROUNDED_CONTEXT_LOCK_INVALID"
                )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            try:
                try:
                    encoded = _read_regular_file_at(
                        root_descriptor,
                        "context_workspace.json",
                    )
                except FileNotFoundError:
                    encoded = b""
                if encoded:
                    try:
                        observed = json.loads(encoded.decode("utf-8"))
                    except (UnicodeError, json.JSONDecodeError) as exc:
                        raise GroundedContextWorkspaceError(
                            "GROUNDED_CONTEXT_MANIFEST_CORRUPT"
                        ) from exc
                    if observed != expected:
                        raise GroundedContextWorkspaceError(
                            "GROUNDED_CONTEXT_REQUEST_CONFLICT"
                        )
                    return
                encoded_manifest = json.dumps(
                    dict(expected),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ).encode("utf-8")
                _atomic_write_at(
                    root_descriptor,
                    "context_workspace.json",
                    encoded_manifest,
                    error_code="GROUNDED_CONTEXT_MANIFEST_WRITE_FAILED",
                )
            finally:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)

    def subagent_workspace(self, kind: str, job_id: str) -> Path:
        components = (
            _path_component(kind, "subagent"),
            _path_component(job_id, "job"),
        )
        try:
            descriptor = _open_directory_beneath(
                self.subagents_root,
                components,
                create=True,
                exclusive_final=True,
            )
            os.close(descriptor)
            return self.subagents_root.joinpath(*components)
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedContextWorkspaceError(
                "GROUNDED_SUBAGENT_WORKSPACE_OUTSIDE_ROOT"
            ) from exc

    def write_core_scratch(self, relative_path: str, content: str) -> Path:
        raw = str(relative_path or "").strip().replace("\\", "/").lstrip("/")
        if raw == "workspace":
            raw = ""
        elif raw.startswith("workspace/"):
            raw = raw[len("workspace/") :]
        components = tuple(
            component
            for component in raw.split("/")
            if component
        )
        if (
            not components
            or any(component in {".", ".."} for component in components)
            or any(component.startswith(".context-") for component in components)
        ):
            raise GroundedContextWorkspaceError(
                "GROUNDED_SCRATCH_PATH_RESERVED"
            )
        root_descriptor = -1
        lock_descriptor = -1
        parent_descriptor = -1
        try:
            root_descriptor = _open_directory_beneath(
                self.core_scratch_root
            )
            lock_descriptor = os.open(
                ".context-scratch.lock",
                os.O_RDWR
                | os.O_CREAT
                | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=root_descriptor,
            )
            if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
                raise GroundedContextWorkspaceError(
                    "GROUNDED_SCRATCH_LOCK_INVALID"
                )
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
            try:
                parent_descriptor = _open_directory_beneath(
                    self.core_scratch_root,
                    components[:-1],
                    create=True,
                )
                _atomic_write_at(
                    parent_descriptor,
                    components[-1],
                    str(content or "").encode("utf-8"),
                    error_code="GROUNDED_SCRATCH_WRITE_FAILED",
                )
            finally:
                fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        except (OSError, GroundedContextWorkspaceError) as exc:
            raise GroundedContextWorkspaceError(
                "GROUNDED_SCRATCH_WRITE_FAILED"
            ) from exc
        finally:
            if parent_descriptor >= 0:
                os.close(parent_descriptor)
            if lock_descriptor >= 0:
                os.close(lock_descriptor)
            if root_descriptor >= 0:
                os.close(root_descriptor)
        return self.core_scratch_root.joinpath(*components)
