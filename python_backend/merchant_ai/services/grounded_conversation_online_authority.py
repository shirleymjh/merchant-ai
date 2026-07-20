from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import ValidationError

from merchant_ai.models import ResultCoverage
from merchant_ai.services.grounded_conversation_semantic_resolver import (
    ConversationReferentCandidate,
    ConversationSemanticProvider,
    build_conversation_semantic_resolver_request,
    review_conversation_semantics,
)
from merchant_ai.services.grounded_conversation_state import (
    GroundedConversationResolution,
    resolve_grounded_conversation_turn,
)
from merchant_ai.services.grounded_query_contract import GroundedQueryContract
from merchant_ai.services.grounded_sql_candidate import (
    grounded_query_contract_fingerprint,
)


class ConversationOnlineAuthorityCode(str, Enum):
    SNAPSHOT_INVALID = "CONVERSATION_AUTHORITY_SNAPSHOT_INVALID"
    PRINCIPAL_MISMATCH = "CONVERSATION_AUTHORITY_PRINCIPAL_MISMATCH"
    ARTIFACT_INDEX_INVALID = "CONVERSATION_AUTHORITY_ARTIFACT_INDEX_INVALID"
    SOURCE_RECORD_INVALID = "CONVERSATION_AUTHORITY_SOURCE_RECORD_INVALID"
    SOURCE_ARTIFACT_CONFLICT = "CONVERSATION_AUTHORITY_SOURCE_ARTIFACT_CONFLICT"
    PUBLICATION_REQUIRED = "CONVERSATION_AUTHORITY_PUBLICATION_REQUIRED"
    RECEIPT_INVALID = "CONVERSATION_AUTHORITY_RECEIPT_INVALID"
    PATH_INVALID = "CONVERSATION_AUTHORITY_PATH_INVALID"
    WORKSPACE_INVALID = "CONVERSATION_AUTHORITY_WORKSPACE_INVALID"
    IMMUTABLE_MARKER_INVALID = "CONVERSATION_AUTHORITY_IMMUTABLE_MARKER_INVALID"
    ARTIFACT_BYTES_INVALID = "CONVERSATION_AUTHORITY_ARTIFACT_BYTES_INVALID"
    MANIFEST_INVALID = "CONVERSATION_AUTHORITY_MANIFEST_INVALID"
    MANIFEST_BINDING_MISMATCH = (
        "CONVERSATION_AUTHORITY_MANIFEST_BINDING_MISMATCH"
    )
    ROWS_BINDING_MISMATCH = "CONVERSATION_AUTHORITY_ROWS_BINDING_MISMATCH"
    ROWS_INVALID = "CONVERSATION_AUTHORITY_ROWS_INVALID"
    CONTRACT_INVALID = "CONVERSATION_AUTHORITY_CONTRACT_INVALID"
    CONTRACT_BINDING_MISMATCH = (
        "CONVERSATION_AUTHORITY_CONTRACT_BINDING_MISMATCH"
    )
    SQL_BINDING_MISMATCH = "CONVERSATION_AUTHORITY_SQL_BINDING_MISMATCH"
    OWNER_MISMATCH = "CONVERSATION_AUTHORITY_OWNER_MISMATCH"
    ACTIVATION_MISMATCH = "CONVERSATION_AUTHORITY_ACTIVATION_MISMATCH"


class GroundedConversationOnlineAuthorityError(RuntimeError):
    def __init__(self, code: ConversationOnlineAuthorityCode) -> None:
        self.code = code
        super().__init__(code.value)


@dataclass(frozen=True)
class _VerifiedPublication:
    artifact_fingerprint: str
    rows_sha256: str
    rows_content_address: str
    result_coverage: str
    result_is_truncated: bool


def _fail(code: ConversationOnlineAuthorityCode) -> None:
    raise GroundedConversationOnlineAuthorityError(code)


def _text(value: Any) -> str:
    return str(value or "").strip()


def _valid_sha256(value: Any) -> bool:
    normalized = _text(value)
    return len(normalized) == 64 and all(
        character in "0123456789abcdef" for character in normalized
    )


def _stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def grounded_conversation_authority_fingerprint(
    authority_name: str,
    deployment: Mapping[str, Any],
) -> str:
    normalized_name = _text(authority_name)
    if not normalized_name:
        raise ValueError("authority_name is required")
    return _stable_fingerprint(
        {
            "authorityName": normalized_name,
            "deployment": dict(deployment or {}),
        }
    )


def _strict_json_object(encoded: bytes) -> dict[str, Any]:
    def reject_duplicate_keys(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise ValueError("duplicate JSON object key")
            output[key] = value
        return output

    value = json.loads(
        encoded.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
    )
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object")
    return value


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return tuple(value)
    _fail(ConversationOnlineAuthorityCode.SNAPSHOT_INVALID)


def _unique_text(values: Sequence[Any]) -> tuple[str, ...]:
    normalized = tuple(_text(value) for value in values)
    if any(not value for value in normalized):
        _fail(ConversationOnlineAuthorityCode.ARTIFACT_INDEX_INVALID)
    if len(set(normalized)) != len(normalized):
        _fail(ConversationOnlineAuthorityCode.SOURCE_ARTIFACT_CONFLICT)
    return normalized


def _relative_components(value: Any) -> tuple[str, ...]:
    path = _text(value)
    if not path or path.startswith("/") or "\\" in path:
        _fail(ConversationOnlineAuthorityCode.PATH_INVALID)
    components = tuple(path.split("/"))
    if any(
        not component
        or component in {".", ".."}
        or "/" in component
        or "\\" in component
        for component in components
    ):
        _fail(ConversationOnlineAuthorityCode.PATH_INVALID)
    return components


def _directory_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_mode,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_mode,
        right.st_size,
        right.st_mtime_ns,
    )


def _immutable_marker_name(file_name: str) -> str:
    return ".artifact-immutable-%s.sha256" % hashlib.sha256(
        file_name.encode("utf-8")
    ).hexdigest()


class _JsonObjectArrayCounter:
    """Count a publisher-produced JSON row array without retaining rows."""

    _WHITESPACE = {9, 10, 13, 32}

    def __init__(self) -> None:
        self.stack: list[int] = []
        self.started = False
        self.finished = False
        self.in_string = False
        self.escape = False
        self.expect_value = True
        self.row_count = 0

    def consume(self, chunk: bytes) -> None:
        for character in chunk:
            if self.finished:
                if character not in self._WHITESPACE:
                    _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
                continue
            if not self.started:
                if character in self._WHITESPACE:
                    continue
                if character != ord("["):
                    _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
                self.started = True
                self.stack.append(character)
                continue
            if self.in_string:
                if self.escape:
                    self.escape = False
                elif character == ord("\\"):
                    self.escape = True
                elif character == ord('"'):
                    self.in_string = False
                continue
            if len(self.stack) == 1:
                if character in self._WHITESPACE:
                    continue
                if self.expect_value:
                    if character == ord("]"):
                        self.stack.pop()
                        self.finished = True
                        continue
                    if character != ord("{"):
                        _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
                    self.stack.append(character)
                    self.row_count += 1
                    self.expect_value = False
                    continue
                if character == ord(","):
                    self.expect_value = True
                    continue
                if character == ord("]"):
                    self.stack.pop()
                    self.finished = True
                    continue
                _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
            if character == ord('"'):
                self.in_string = True
            elif character in {ord("["), ord("{")}:
                self.stack.append(character)
            elif character in {ord("]"), ord("}")}:
                expected = ord("[") if character == ord("]") else ord("{")
                if not self.stack or self.stack[-1] != expected:
                    _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
                self.stack.pop()

    def finish(self) -> int:
        if (
            not self.started
            or not self.finished
            or self.stack
            or self.in_string
            or self.escape
        ):
            _fail(ConversationOnlineAuthorityCode.ROWS_INVALID)
        return self.row_count


class _WorkspacePublicationReader:
    _MAX_MANIFEST_BYTES = 8 * 1024 * 1024

    def __init__(self, workspace_root: Path | str) -> None:
        self.workspace_root = Path(os.path.abspath(str(workspace_root)))

    def open_root(self) -> int:
        try:
            descriptor = os.open(self.workspace_root, _directory_flags())
            if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
                raise OSError("workspace root is not a directory")
            return descriptor
        except OSError as exc:
            _fail(ConversationOnlineAuthorityCode.WORKSPACE_INVALID)
            raise AssertionError from exc

    @staticmethod
    def _open_directory_at(
        root_descriptor: int,
        components: Sequence[str],
    ) -> int:
        descriptor = os.dup(root_descriptor)
        try:
            for component in components:
                child = os.open(
                    component,
                    _directory_flags(),
                    dir_fd=descriptor,
                )
                if not stat.S_ISDIR(os.fstat(child).st_mode):
                    os.close(child)
                    _fail(ConversationOnlineAuthorityCode.PATH_INVALID)
                os.close(descriptor)
                descriptor = child
            return descriptor
        except GroundedConversationOnlineAuthorityError:
            os.close(descriptor)
            raise
        except OSError as exc:
            os.close(descriptor)
            _fail(ConversationOnlineAuthorityCode.PATH_INVALID)
            raise AssertionError from exc

    @staticmethod
    def _read_small_regular(
        parent_descriptor: int,
        name: str,
        *,
        max_bytes: int,
    ) -> tuple[bytes, os.stat_result]:
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                _file_flags(),
                dir_fd=parent_descriptor,
            )
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size > max_bytes:
                _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)
            output = bytearray()
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, max_bytes + 1))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > max_bytes:
                    _fail(
                        ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID
                    )
            return bytes(output), opened
        except GroundedConversationOnlineAuthorityError:
            raise
        except OSError as exc:
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)
            raise AssertionError from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    @staticmethod
    def _assert_named_identity(
        parent_descriptor: int,
        name: str,
        expected: os.stat_result,
    ) -> None:
        try:
            current = os.stat(
                name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)
            raise AssertionError from exc
        if not stat.S_ISREG(current.st_mode) or not _same_identity(
            expected,
            current,
        ):
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)

    def _verified_parent(
        self,
        root_descriptor: int,
        artifact_root: Sequence[str],
        relative_path: Any,
    ) -> tuple[int, str]:
        components = _relative_components(relative_path)
        parent = self._open_directory_at(
            root_descriptor,
            (*artifact_root, *components[:-1]),
        )
        return parent, components[-1]

    def read_manifest(
        self,
        root_descriptor: int,
        artifact_root: Sequence[str],
        relative_path: Any,
        expected_sha256: str,
    ) -> bytes:
        parent, name = self._verified_parent(
            root_descriptor,
            artifact_root,
            relative_path,
        )
        try:
            marker_name = _immutable_marker_name(name)
            marker, marker_identity = self._read_small_regular(
                parent,
                marker_name,
                max_bytes=128,
            )
            try:
                marker_digest = marker.decode("ascii").strip()
            except UnicodeError:
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
            if marker_digest != expected_sha256:
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
            encoded, artifact_identity = self._read_small_regular(
                parent,
                name,
                max_bytes=self._MAX_MANIFEST_BYTES,
            )
            if hashlib.sha256(encoded).hexdigest() != expected_sha256:
                _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)
            self._assert_named_identity(parent, name, artifact_identity)
            marker_again, marker_identity_again = self._read_small_regular(
                parent,
                marker_name,
                max_bytes=128,
            )
            if (
                marker_again != marker
                or not _same_identity(marker_identity, marker_identity_again)
            ):
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
            return encoded
        finally:
            os.close(parent)

    def verify_rows(
        self,
        root_descriptor: int,
        artifact_root: Sequence[str],
        relative_path: Any,
        *,
        expected_sha256: str,
        expected_bytes: int,
        expected_rows: int,
    ) -> None:
        parent, name = self._verified_parent(
            root_descriptor,
            artifact_root,
            relative_path,
        )
        descriptor = -1
        try:
            marker_name = _immutable_marker_name(name)
            marker, marker_identity = self._read_small_regular(
                parent,
                marker_name,
                max_bytes=128,
            )
            try:
                marker_digest = marker.decode("ascii").strip()
            except UnicodeError:
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
            if marker_digest != expected_sha256:
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
            descriptor = os.open(name, _file_flags(), dir_fd=parent)
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode) or opened.st_size != expected_bytes:
                _fail(ConversationOnlineAuthorityCode.ROWS_BINDING_MISMATCH)
            digest = hashlib.sha256()
            counter = _JsonObjectArrayCounter()
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
                counter.consume(chunk)
            if (
                digest.hexdigest() != expected_sha256
                or counter.finish() != expected_rows
            ):
                _fail(ConversationOnlineAuthorityCode.ROWS_BINDING_MISMATCH)
            self._assert_named_identity(parent, name, opened)
            marker_again, marker_identity_again = self._read_small_regular(
                parent,
                marker_name,
                max_bytes=128,
            )
            if (
                marker_again != marker
                or not _same_identity(marker_identity, marker_identity_again)
            ):
                _fail(
                    ConversationOnlineAuthorityCode.IMMUTABLE_MARKER_INVALID
                )
        except GroundedConversationOnlineAuthorityError:
            raise
        except OSError as exc:
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_BYTES_INVALID)
            raise AssertionError from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            os.close(parent)


class PersistedConversationPublishedCandidateReader:
    """Rebuild candidates only from reopened, immutable published artifacts."""

    def __init__(
        self,
        *,
        workspace_root: Path | str,
        expected_principal_fingerprint: str,
        expected_context_owner_fingerprint: str,
        expected_semantic_activation_fingerprint: str,
    ) -> None:
        for value in (
            expected_principal_fingerprint,
            expected_context_owner_fingerprint,
            expected_semantic_activation_fingerprint,
        ):
            if not _valid_sha256(value):
                raise ValueError("authority fingerprints must be sha256 values")
        self.expected_principal_fingerprint = _text(
            expected_principal_fingerprint
        )
        self.expected_context_owner_fingerprint = _text(
            expected_context_owner_fingerprint
        )
        self.expected_semantic_activation_fingerprint = _text(
            expected_semantic_activation_fingerprint
        )
        self.publications = _WorkspacePublicationReader(workspace_root)

    def read_candidates(
        self,
        persisted_snapshot: Mapping[str, Any] | None,
    ) -> tuple[ConversationReferentCandidate, ...]:
        if persisted_snapshot is None:
            return ()
        if not isinstance(persisted_snapshot, Mapping):
            _fail(ConversationOnlineAuthorityCode.SNAPSHOT_INVALID)
        active_scope = persisted_snapshot.get("activeScope")
        if active_scope is None:
            return ()
        if not isinstance(active_scope, Mapping):
            _fail(ConversationOnlineAuthorityCode.SNAPSHOT_INVALID)
        artifact_ids = _unique_text(
            _sequence(active_scope.get("artifactIds") or ())
        )
        records = _sequence(active_scope.get("sourceArtifacts") or ())
        if not artifact_ids and not records:
            return ()
        if _text(persisted_snapshot.get("principalFingerprint")) != (
            self.expected_principal_fingerprint
        ):
            _fail(ConversationOnlineAuthorityCode.PRINCIPAL_MISMATCH)
        if len(artifact_ids) != len(records):
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_INDEX_INVALID)

        indexed: dict[str, Mapping[str, Any]] = {}
        for raw_record in records:
            if not isinstance(raw_record, Mapping):
                _fail(ConversationOnlineAuthorityCode.SOURCE_RECORD_INVALID)
            artifact_id = _text(raw_record.get("queryArtifactId"))
            if not artifact_id:
                _fail(ConversationOnlineAuthorityCode.SOURCE_RECORD_INVALID)
            if artifact_id in indexed:
                _fail(
                    ConversationOnlineAuthorityCode.SOURCE_ARTIFACT_CONFLICT
                )
            indexed[artifact_id] = raw_record
        if set(indexed) != set(artifact_ids):
            _fail(ConversationOnlineAuthorityCode.ARTIFACT_INDEX_INVALID)

        root_descriptor = self.publications.open_root()
        publications: set[str] = set()
        candidates: list[ConversationReferentCandidate] = []
        try:
            for artifact_id in artifact_ids:
                candidate, publication = self._candidate_from_record(
                    root_descriptor,
                    artifact_id,
                    indexed[artifact_id],
                )
                if publication.artifact_fingerprint in publications:
                    _fail(
                        ConversationOnlineAuthorityCode.SOURCE_ARTIFACT_CONFLICT
                    )
                publications.add(publication.artifact_fingerprint)
                candidates.append(candidate)
        finally:
            os.close(root_descriptor)
        return tuple(candidates)

    @staticmethod
    def _receipt(record: Mapping[str, Any]) -> Mapping[str, Any]:
        singular_present = "publicationReceipt" in record
        plural_present = "resultArtifactReceipts" in record
        if singular_present == plural_present:
            _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        if singular_present:
            receipt = record.get("publicationReceipt")
        else:
            receipts = _sequence(record.get("resultArtifactReceipts"))
            if len(receipts) != 1:
                _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
            receipt = receipts[0]
        if not isinstance(receipt, Mapping):
            _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        return receipt

    def _candidate_from_record(
        self,
        root_descriptor: int,
        artifact_id: str,
        record: Mapping[str, Any],
    ) -> tuple[ConversationReferentCandidate, _VerifiedPublication]:
        if _text(record.get("publicationStatus")) != "PUBLISHED":
            _fail(ConversationOnlineAuthorityCode.PUBLICATION_REQUIRED)
        receipt = self._receipt(record)
        receipt_artifact_id = receipt.get("queryArtifactId")
        if receipt_artifact_id is not None and _text(receipt_artifact_id) != (
            artifact_id
        ):
            _fail(ConversationOnlineAuthorityCode.SOURCE_ARTIFACT_CONFLICT)
        raw_contract = record.get("contract")
        if not isinstance(raw_contract, Mapping):
            _fail(ConversationOnlineAuthorityCode.CONTRACT_INVALID)
        try:
            contract = GroundedQueryContract.model_validate(raw_contract)
        except (TypeError, ValueError, ValidationError) as exc:
            _fail(ConversationOnlineAuthorityCode.CONTRACT_INVALID)
            raise AssertionError from exc
        if not contract.ready or contract.status != "READY":
            _fail(ConversationOnlineAuthorityCode.CONTRACT_INVALID)
        contract_fingerprint = grounded_query_contract_fingerprint(contract)
        stored_contract_fingerprint = _text(
            record.get("contractFingerprint")
        )
        stored_sql_fingerprint = _text(record.get("sqlFingerprint"))
        if (
            stored_contract_fingerprint != contract_fingerprint
            or not _valid_sha256(stored_contract_fingerprint)
        ):
            _fail(
                ConversationOnlineAuthorityCode.CONTRACT_BINDING_MISMATCH
            )
        if not _valid_sha256(stored_sql_fingerprint):
            _fail(ConversationOnlineAuthorityCode.SQL_BINDING_MISMATCH)
        artifact_root = _relative_components(
            record.get("artifactRootRelativePath")
        )
        publication = self._verify_publication(
            root_descriptor,
            artifact_root,
            artifact_id,
            receipt,
            contract_fingerprint=contract_fingerprint,
            sql_fingerprint=stored_sql_fingerprint,
        )
        return (
            self._project_candidate(
                artifact_id,
                contract,
                publication,
                contract_fingerprint=contract_fingerprint,
                sql_fingerprint=stored_sql_fingerprint,
            ),
            publication,
        )

    def _verify_publication(
        self,
        root_descriptor: int,
        artifact_root: Sequence[str],
        artifact_id: str,
        receipt: Mapping[str, Any],
        *,
        contract_fingerprint: str,
        sql_fingerprint: str,
    ) -> _VerifiedPublication:
        digest_fields = (
            "artifactFingerprint",
            "queryManifestSha256",
            "rowsSha256",
            "sqlSha256",
            "contractFingerprint",
            "sqlEvidenceFingerprint",
            "dataSnapshotFingerprint",
            "verifiedEvidenceSha256",
            "attemptFingerprint",
        )
        if any(not _valid_sha256(receipt.get(key)) for key in digest_fields):
            _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        for address_key, digest_key in (
            ("manifestContentAddress", "queryManifestSha256"),
            ("rowsContentAddress", "rowsSha256"),
            ("sqlContentAddress", "sqlSha256"),
        ):
            if _text(receipt.get(address_key)) != "sha256:%s" % _text(
                receipt.get(digest_key)
            ):
                _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        if receipt.get("publicationStatus") not in {None, "PUBLISHED"}:
            _fail(ConversationOnlineAuthorityCode.PUBLICATION_REQUIRED)
        if _text(receipt.get("contractFingerprint")) != contract_fingerprint:
            _fail(
                ConversationOnlineAuthorityCode.CONTRACT_BINDING_MISMATCH
            )
        if _text(receipt.get("sqlEvidenceFingerprint")) != sql_fingerprint:
            _fail(ConversationOnlineAuthorityCode.SQL_BINDING_MISMATCH)
        if _text(receipt.get("contextOwnerFingerprint")) != (
            self.expected_context_owner_fingerprint
        ):
            _fail(ConversationOnlineAuthorityCode.OWNER_MISMATCH)
        if _text(receipt.get("semanticActivationFingerprint")) != (
            self.expected_semantic_activation_fingerprint
        ):
            _fail(ConversationOnlineAuthorityCode.ACTIVATION_MISMATCH)
        for key in (
            "storedRowCount",
            "artifactRowCount",
            "artifactByteCount",
            "exactResultRowCount",
            "executionGeneration",
        ):
            value = receipt.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        if (
            int(receipt.get("executionGeneration") or 0) <= 0
            or not isinstance(receipt.get("resultIsTruncated"), bool)
            or receipt.get("artifactComplete") is not True
            or receipt.get("artifactCoverage") != "ALL_ROWS"
            or receipt.get("artifactRowCount")
            != receipt.get("exactResultRowCount")
            or int(receipt.get("storedRowCount") or 0)
            > int(receipt.get("exactResultRowCount") or 0)
        ):
            _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)
        coverage = _text(receipt.get("resultCoverage"))
        if coverage not in {item.value for item in ResultCoverage}:
            _fail(ConversationOnlineAuthorityCode.RECEIPT_INVALID)

        manifest_sha = _text(receipt.get("queryManifestSha256"))
        manifest_encoded = self.publications.read_manifest(
            root_descriptor,
            artifact_root,
            receipt.get("manifestRelativePath"),
            manifest_sha,
        )
        try:
            manifest = _strict_json_object(manifest_encoded)
        except (UnicodeError, ValueError, json.JSONDecodeError) as exc:
            _fail(ConversationOnlineAuthorityCode.MANIFEST_INVALID)
            raise AssertionError from exc
        expected_scalars = {
            "schemaVersion": 3,
            "artifactKind": "GROUNDED_QUERY_RESULT",
            "publicationStatus": "VERIFIED",
            "artifactFingerprint": receipt.get("artifactFingerprint"),
            "contractFingerprint": contract_fingerprint,
            "sqlEvidenceFingerprint": sql_fingerprint,
            "sqlSha256": receipt.get("sqlSha256"),
            "contextOwnerFingerprint": (
                self.expected_context_owner_fingerprint
            ),
            "semanticActivationFingerprint": (
                self.expected_semantic_activation_fingerprint
            ),
            "resultCoverage": receipt.get("resultCoverage"),
            "resultIsTruncated": receipt.get("resultIsTruncated"),
            "storedRowCount": receipt.get("storedRowCount"),
            "artifactRowCount": receipt.get("artifactRowCount"),
            "artifactByteCount": receipt.get("artifactByteCount"),
            "artifactCoverage": "ALL_ROWS",
            "artifactComplete": True,
            "exactResultRowCount": receipt.get("exactResultRowCount"),
            "executionGeneration": receipt.get("executionGeneration"),
        }
        if any(
            manifest.get(key) != value
            for key, value in expected_scalars.items()
        ):
            _fail(
                ConversationOnlineAuthorityCode.MANIFEST_BINDING_MISMATCH
            )
        attempt_id = _text(manifest.get("executionAttemptId"))
        if (
            not attempt_id
            or hashlib.sha256(attempt_id.encode("utf-8")).hexdigest()
            != receipt.get("attemptFingerprint")
        ):
            _fail(
                ConversationOnlineAuthorityCode.MANIFEST_BINDING_MISMATCH
            )
        verified_evidence = manifest.get("verifiedEvidence")
        data_snapshot = manifest.get("dataSnapshot")
        if (
            not isinstance(verified_evidence, Mapping)
            or not bool(verified_evidence.get("passed"))
            or _stable_fingerprint(verified_evidence)
            != receipt.get("verifiedEvidenceSha256")
            or manifest.get("verifiedEvidenceSha256")
            != receipt.get("verifiedEvidenceSha256")
            or not isinstance(data_snapshot, Mapping)
            or _stable_fingerprint(data_snapshot)
            != receipt.get("dataSnapshotFingerprint")
            or _text(data_snapshot.get("semanticActivationFingerprint"))
            != self.expected_semantic_activation_fingerprint
        ):
            _fail(
                ConversationOnlineAuthorityCode.MANIFEST_BINDING_MISMATCH
            )
        if manifest.get("queryArtifactId") not in {None, artifact_id}:
            _fail(ConversationOnlineAuthorityCode.SOURCE_ARTIFACT_CONFLICT)

        rows_reference = manifest.get("rowsArtifact")
        sql_reference = manifest.get("sqlArtifact")
        if not isinstance(rows_reference, Mapping) or not isinstance(
            sql_reference,
            Mapping,
        ):
            _fail(ConversationOnlineAuthorityCode.MANIFEST_INVALID)
        rows_path = "/".join(
            _relative_components(rows_reference.get("relativePath"))
        )
        receipt_rows_path = "/".join(
            _relative_components(receipt.get("rowsRelativePath"))
        )
        sql_path = "/".join(
            _relative_components(sql_reference.get("relativePath"))
        )
        receipt_sql_path = "/".join(
            _relative_components(receipt.get("sqlRelativePath"))
        )
        if (
            rows_path != receipt_rows_path
            or rows_reference.get("sha256") != receipt.get("rowsSha256")
            or rows_reference.get("contentAddress")
            != receipt.get("rowsContentAddress")
            or rows_reference.get("bytes")
            != receipt.get("artifactByteCount")
            or rows_reference.get("immutable") is not True
        ):
            _fail(ConversationOnlineAuthorityCode.ROWS_BINDING_MISMATCH)
        if (
            sql_path != receipt_sql_path
            or sql_reference.get("sha256") != receipt.get("sqlSha256")
            or sql_reference.get("contentAddress")
            != receipt.get("sqlContentAddress")
            or sql_reference.get("immutable") is not True
        ):
            _fail(ConversationOnlineAuthorityCode.SQL_BINDING_MISMATCH)
        self.publications.verify_rows(
            root_descriptor,
            artifact_root,
            rows_path,
            expected_sha256=_text(receipt.get("rowsSha256")),
            expected_bytes=int(receipt.get("artifactByteCount") or 0),
            expected_rows=int(receipt.get("artifactRowCount") or 0),
        )
        return _VerifiedPublication(
            artifact_fingerprint=_text(receipt.get("artifactFingerprint")),
            rows_sha256=_text(receipt.get("rowsSha256")),
            rows_content_address=_text(receipt.get("rowsContentAddress")),
            result_coverage=coverage,
            result_is_truncated=bool(receipt.get("resultIsTruncated")),
        )

    @staticmethod
    def _project_candidate(
        artifact_id: str,
        contract: GroundedQueryContract,
        publication: _VerifiedPublication,
        *,
        contract_fingerprint: str,
        sql_fingerprint: str,
    ) -> ConversationReferentCandidate:
        topics = tuple(dict.fromkeys(_text(item) for item in contract.topics if _text(item)))
        tables = tuple(
            dict.fromkeys(
                _text(item.table) for item in contract.tables if _text(item.table)
            )
        )
        entity_identities = tuple(
            dict.fromkeys(
                _text(getattr(item, "entity_identity", ""))
                for item in (
                    *contract.dimensions,
                    *contract.selected_fields,
                    *contract.entity_filters,
                )
                if _text(getattr(item, "entity_identity", ""))
            )
        )
        grains = tuple(
            dict.fromkeys(
                _text(item.data_grain)
                for item in contract.tables
                if _text(item.data_grain)
            )
        )
        time_range = contract.time_range
        time_values = tuple(
            dict.fromkeys(
                value
                for value in (
                    _text(time_range.execution_start_date or time_range.start_date),
                    _text(time_range.execution_end_date or time_range.end_date),
                    _text(time_range.label),
                )
                if value
            )
        )
        filter_values = tuple(
            dict.fromkeys(
                _text(item.requested_phrase)
                or "%s.%s:%s"
                % (_text(item.table), _text(item.column), _text(item.operator))
                for item in contract.entity_filters
            )
        )
        complete_membership = bool(
            publication.result_coverage
            in {ResultCoverage.ALL_ROWS.value, ResultCoverage.TOP_N.value}
            and not publication.result_is_truncated
        )
        return ConversationReferentCandidate(
            artifact_id=artifact_id,
            contract_fingerprint=contract_fingerprint,
            sql_fingerprint=sql_fingerprint,
            query_shape=_text(contract.query_shape) or "UNSPECIFIED",
            coverage_status=publication.result_coverage,
            # The governed query question gives the semantic reviewer the
            # previous-turn language context while the artifact id and
            # fingerprints remain the only executable authority.
            label=(
                _text(contract.question)
                or "%s:%s" % (_text(contract.query_shape), artifact_id)
            ),
            topic_ids=topics,
            table_ids=tables,
            entity_identities=entity_identities,
            data_grains=grains,
            time_scope_labels=time_values,
            filter_scope_labels=filter_values,
            membership_handle_type=(
                "PUBLISHED_RESULT_ROWS" if complete_membership else ""
            ),
            membership_handle_id=(
                publication.rows_content_address if complete_membership else ""
            ),
            membership_values_hash=(
                publication.rows_sha256 if complete_membership else ""
            ),
            snapshot_semantics="IMMUTABLE_PUBLISHED_RESULT_SNAPSHOT",
        )


class GroundedConversationOnlineAuthorityFacade:
    """Narrow safety boundary for one Core conversation-resolution call."""

    def __init__(
        self,
        *,
        workspace_root: Path | str,
        semantic_provider: ConversationSemanticProvider | None,
        trusted_reviewer_authority_fingerprints: Sequence[str],
        core_authority_fingerprint: str,
        review_timeout_seconds: float = 8.0,
    ) -> None:
        normalized_workspace_root = Path(
            os.path.abspath(str(workspace_root or ""))
        )
        if not str(workspace_root or "").strip():
            raise ValueError("workspace_root is required")
        trusted = tuple(
            dict.fromkeys(
                _text(value)
                for value in trusted_reviewer_authority_fingerprints
                if _text(value)
            )
        )
        if semantic_provider is not None and not trusted:
            raise ValueError("trusted reviewer authorities are required")
        if review_timeout_seconds <= 0:
            raise ValueError("review_timeout_seconds must be positive")
        self.workspace_root = normalized_workspace_root
        self.semantic_provider = semantic_provider
        self.trusted_reviewer_authority_fingerprints = trusted
        self.core_authority_fingerprint = _text(core_authority_fingerprint)
        self.review_timeout_seconds = float(review_timeout_seconds)

    def resolve(
        self,
        question: str,
        *,
        persisted_snapshot: Mapping[str, Any] | None,
        persisted_revision: int = 0,
        request_context: Any = None,
        expected_principal_fingerprint: str,
        expected_context_owner_fingerprint: str,
        expected_semantic_activation_fingerprint: str = "",
    ) -> GroundedConversationResolution:
        normalized_question = _text(question)
        snapshot = dict(persisted_snapshot or {})
        active_scope = snapshot.get("activeScope")
        has_retained_artifacts = bool(
            isinstance(active_scope, Mapping)
            and (
                active_scope.get("artifactIds")
                or active_scope.get("sourceArtifacts")
            )
        )
        try:
            if has_retained_artifacts:
                candidate_reader = (
                    PersistedConversationPublishedCandidateReader(
                        workspace_root=self.workspace_root,
                        expected_principal_fingerprint=(
                            expected_principal_fingerprint
                        ),
                        expected_context_owner_fingerprint=(
                            expected_context_owner_fingerprint
                        ),
                        expected_semantic_activation_fingerprint=(
                            expected_semantic_activation_fingerprint
                        ),
                    )
                )
                candidates = candidate_reader.read_candidates(snapshot)
            else:
                candidates = ()
        except (
            GroundedConversationOnlineAuthorityError,
            TypeError,
            ValueError,
        ) as exc:
            issue_code = (
                exc.code.value
                if isinstance(
                    exc,
                    GroundedConversationOnlineAuthorityError,
                )
                else ConversationOnlineAuthorityCode.SNAPSHOT_INVALID.value
            )
            return GroundedConversationResolution(
                original_question=normalized_question,
                effective_question=normalized_question,
                status="PUBLISHED_CONTEXT_AUTHORITY_REJECTED",
                source_revision=max(0, int(persisted_revision or 0)),
                source="PUBLISHED_CONVERSATION_AUTHORITY",
                clarification_question=(
                    "服务端保留的结果制品未通过完整性复核。"
                    "请明确说明本轮所需的数据范围。"
                ),
                clarification_type="CONVERSATION_PUBLISHED_CONTEXT_INVALID",
                semantic_issue_codes=(issue_code,),
            )

        request = build_conversation_semantic_resolver_request(
            normalized_question,
            candidates,
        )
        semantic_review = None
        if (
            (candidates or has_retained_artifacts)
            and self.semantic_provider is not None
        ):
            semantic_review = review_conversation_semantics(
                self.semantic_provider,
                request,
                trusted_authority_fingerprints=(
                    self.trusted_reviewer_authority_fingerprints
                ),
                core_authority_fingerprint=self.core_authority_fingerprint,
                timeout_seconds=self.review_timeout_seconds,
            )
        return resolve_grounded_conversation_turn(
            normalized_question,
            semantic_review=semantic_review,
            verified_candidates=candidates,
            persisted_snapshot=persisted_snapshot,
            persisted_revision=persisted_revision,
            request_context=request_context,
        )


__all__ = [
    "ConversationOnlineAuthorityCode",
    "GroundedConversationOnlineAuthorityError",
    "GroundedConversationOnlineAuthorityFacade",
    "PersistedConversationPublishedCandidateReader",
    "grounded_conversation_authority_fingerprint",
]
