from __future__ import annotations

import hashlib
import json
import sys
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import DataSnapshotContract
from merchant_ai.services import repositories
from merchant_ai.services import grounded_result_streaming as streaming_module
from merchant_ai.services.grounded_result_streaming import (
    GroundedResultArtifactReceipt,
    GroundedResultCoverage,
    GroundedResultStreamCode,
    GroundedResultStreamLimits,
    GroundedResultStreamMaterializer,
    GroundedResultStreamingError,
    grounded_canonical_json_sha256,
)
from merchant_ai.services.repositories import (
    DatabaseClient,
    DatabaseStreamError,
    DorisRepository,
)


def limits(
    *,
    preview_rows: int = 2,
    fetch_batch_rows: int = 3,
    max_rows: int = 100,
    max_bytes: int = 1024 * 1024,
) -> GroundedResultStreamLimits:
    return GroundedResultStreamLimits(
        preview_rows=preview_rows,
        fetch_batch_rows=fetch_batch_rows,
        max_rows=max_rows,
        max_bytes=max_bytes,
    )


def immutable_marker_name(file_name: str) -> str:
    return ".artifact-immutable-%s.sha256" % hashlib.sha256(
        file_name.encode("utf-8")
    ).hexdigest()


def assert_inactive_failure(
    error: GroundedResultStreamingError,
    expected_code: GroundedResultStreamCode,
) -> None:
    assert error.code == expected_code
    assert error.partial.coverage == GroundedResultCoverage.INACTIVE_PARTIAL.value
    assert error.partial.complete is False
    assert error.partial.active is False


def test_streams_canonical_json_with_exact_digest_count_and_bounded_preview(
    tmp_path: Path,
) -> None:
    rows = [
        {"z": 1, "a": "订单一"},
        {"z": 2, "a": "订单二"},
        {"z": 3, "a": "订单三"},
    ]
    receipt = GroundedResultStreamMaterializer(tmp_path).materialize_batches(
        (rows[:1], rows[1:]),
        artifact_id="result-a",
        limits=limits(preview_rows=2),
    )

    expected = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    rows_path = tmp_path / receipt.rows_relative_path
    marker_path = tmp_path / receipt.marker_relative_path

    assert rows_path.read_bytes() == expected
    assert receipt.rows_canonical_sha256 == hashlib.sha256(expected).hexdigest()
    assert receipt.content_address == "sha256:%s" % receipt.rows_canonical_sha256
    assert receipt.byte_count == len(expected)
    assert receipt.exact_row_count == 3
    assert receipt.preview_rows == (
        {"a": "订单一", "z": 1},
        {"a": "订单二", "z": 2},
    )
    assert receipt.preview_is_truncated is True
    assert receipt.coverage == GroundedResultCoverage.ALL_ROWS.value
    assert receipt.complete is True
    assert receipt.active is True
    assert receipt.immutable is True
    assert marker_path.read_text(encoding="ascii").strip() == receipt.rows_canonical_sha256
    assert rows_path.stat().st_mode & 0o777 == 0o400
    assert marker_path.stat().st_mode & 0o777 == 0o400

    from merchant_ai.services.grounded_query_executor import (
        GroundedQueryExecutionKernel,
    )

    assert (
        receipt.rows_canonical_sha256
        == GroundedQueryExecutionKernel._canonical_json_sha256(rows)
    )
    assert receipt.rows_canonical_sha256 == grounded_canonical_json_sha256(rows)


def test_empty_eof_is_a_complete_two_byte_json_array(tmp_path: Path) -> None:
    receipt = GroundedResultStreamMaterializer(tmp_path).materialize_batches(
        (),
        artifact_id="empty-result",
        limits=limits(preview_rows=0),
    )

    assert (tmp_path / receipt.rows_relative_path).read_bytes() == b"[]"
    assert receipt.byte_count == 2
    assert receipt.exact_row_count == 0
    assert receipt.preview_rows == ()
    assert receipt.preview_is_truncated is False
    assert receipt.coverage == GroundedResultCoverage.ALL_ROWS.value


def test_large_row_count_keeps_only_configured_preview(tmp_path: Path) -> None:
    batch_size = 41
    row_total = 25_000

    def batches() -> Iterator[list[dict[str, Any]]]:
        for start in range(0, row_total, batch_size):
            yield [
                {"position": position, "value": "value-%d" % position}
                for position in range(start, min(row_total, start + batch_size))
            ]

    receipt = GroundedResultStreamMaterializer(tmp_path).materialize_batches(
        batches(),
        artifact_id="large-result",
        limits=limits(
            preview_rows=3,
            fetch_batch_rows=batch_size,
            max_rows=row_total,
            max_bytes=8 * 1024 * 1024,
        ),
    )

    assert receipt.exact_row_count == row_total
    assert len(receipt.preview_rows) == 3
    assert receipt.preview_is_truncated is True
    with (tmp_path / receipt.rows_relative_path).open("rb") as stream:
        assert stream.read(1) == b"["
        stream.seek(-1, 2)
        assert stream.read(1) == b"]"


@pytest.mark.parametrize(
    ("stream_limits", "expected_code"),
    [
        (
            limits(max_rows=2),
            GroundedResultStreamCode.ROW_QUOTA_EXCEEDED,
        ),
        (
            limits(max_bytes=10),
            GroundedResultStreamCode.BYTE_QUOTA_EXCEEDED,
        ),
    ],
)
def test_quota_failure_never_activates_partial_artifact(
    tmp_path: Path,
    stream_limits: GroundedResultStreamLimits,
    expected_code: GroundedResultStreamCode,
) -> None:
    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            [[{"value": "one"}, {"value": "two"}, {"value": "three"}]],
            artifact_id="quota-result",
            limits=stream_limits,
        )

    assert_inactive_failure(captured.value, expected_code)
    assert not (tmp_path / "quota-result" / "rows.json").exists()
    assert not (
        tmp_path
        / "quota-result"
        / immutable_marker_name("rows.json")
    ).exists()


def test_cancellation_closes_source_and_removes_partial_bytes(tmp_path: Path) -> None:
    cancel = threading.Event()
    source_closed = threading.Event()

    def batches() -> Iterator[list[dict[str, Any]]]:
        try:
            yield [{"position": 1}]
            cancel.set()
            yield [{"position": 2}]
        finally:
            source_closed.set()

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            batches(),
            artifact_id="cancelled-result",
            limits=limits(),
            cancel_events=[cancel],
        )

    assert_inactive_failure(captured.value, GroundedResultStreamCode.CANCELLED)
    assert captured.value.partial.row_count == 1
    assert source_closed.is_set()
    assert not (tmp_path / "cancelled-result" / "rows.json").exists()


def test_source_exception_is_typed_and_cannot_claim_all_rows(tmp_path: Path) -> None:
    def batches() -> Iterator[list[dict[str, Any]]]:
        yield [{"position": 1}]
        raise LookupError("source stopped before EOF")

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            batches(),
            artifact_id="failed-result",
            limits=limits(),
        )

    assert_inactive_failure(captured.value, GroundedResultStreamCode.SOURCE_FAILED)
    assert captured.value.cause_type == "LookupError"
    assert captured.value.partial.row_count == 1
    assert not (tmp_path / "failed-result" / "rows.json").exists()


@pytest.mark.parametrize(("value", "artifact_id"), [(float("nan"), "nan"), (float("inf"), "inf")])
def test_non_finite_numbers_are_rejected_as_non_canonical_rows(
    tmp_path: Path,
    value: float,
    artifact_id: str,
) -> None:
    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            [[{"value": value}]],
            artifact_id=artifact_id,
            limits=limits(),
        )

    assert_inactive_failure(captured.value, GroundedResultStreamCode.ROW_INVALID)
    assert not (tmp_path / artifact_id / "rows.json").exists()


def test_prepositioned_namespace_symlink_is_rejected_without_following(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "symlink-result").symlink_to(outside, target_is_directory=True)

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            [[{"position": 1}]],
            artifact_id="symlink-result",
            limits=limits(),
        )

    assert_inactive_failure(
        captured.value,
        GroundedResultStreamCode.ARTIFACT_CONFLICT,
    )
    assert list(outside.iterdir()) == []


def test_symlinked_artifact_root_is_rejected_without_writing(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual-root"
    actual_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(actual_root, target_is_directory=True)

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(linked_root).materialize_batches(
            [[{"position": 1}]],
            artifact_id="root-link-result",
            limits=limits(),
        )

    assert_inactive_failure(captured.value, GroundedResultStreamCode.ROOT_INVALID)
    assert list(actual_root.iterdir()) == []


def test_prepositioned_rows_symlink_is_never_overwritten(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("trusted", encoding="utf-8")

    def batches() -> Iterator[list[dict[str, Any]]]:
        yield [{"position": 1}]
        (tmp_path / "rows-link-result" / "rows.json").symlink_to(outside)

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            batches(),
            artifact_id="rows-link-result",
            limits=limits(),
        )

    assert_inactive_failure(
        captured.value,
        GroundedResultStreamCode.PUBLICATION_CONFLICT,
    )
    assert outside.read_text(encoding="utf-8") == "trusted"
    assert not (
        tmp_path
        / "rows-link-result"
        / immutable_marker_name("rows.json")
    ).exists()


def test_namespace_replacement_during_stream_fails_closed(tmp_path: Path) -> None:
    def batches() -> Iterator[list[dict[str, Any]]]:
        yield [{"position": 1}]
        (tmp_path / "swapped-result").rename(tmp_path / "moved-result")
        (tmp_path / "swapped-result").mkdir()

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            batches(),
            artifact_id="swapped-result",
            limits=limits(),
        )

    assert_inactive_failure(
        captured.value,
        GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
    )
    assert not (tmp_path / "swapped-result" / "rows.json").exists()
    assert not (tmp_path / "moved-result" / "rows.json").exists()


@pytest.mark.parametrize("tamper_target", ["rows", "marker"])
def test_final_content_and_marker_are_rehashed_before_receipt_is_signed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper_target: str,
) -> None:
    real_link = streaming_module.os.link
    marker_name = immutable_marker_name("rows.json")

    def tampering_link(source: str, target: str, *args: Any, **kwargs: Any) -> None:
        real_link(source, target, *args, **kwargs)
        if target != marker_name:
            return
        if tamper_target == "rows":
            victim = tmp_path / "tampered-result" / "rows.json"
            victim.chmod(0o600)
            victim.write_bytes(b'[{"position":9}]')
            victim.chmod(0o400)
        else:
            victim = tmp_path / "tampered-result" / marker_name
            victim.chmod(0o600)
            victim.write_bytes(("0" * 64 + "\n").encode("ascii"))
            victim.chmod(0o400)

    monkeypatch.setattr(streaming_module.os, "link", tampering_link)
    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_batches(
            [[{"position": 1}]],
            artifact_id="tampered-result",
            limits=limits(),
        )

    assert_inactive_failure(
        captured.value,
        GroundedResultStreamCode.ARTIFACT_BINDING_CHANGED,
    )
    assert not (tmp_path / "tampered-result" / marker_name).exists()
    assert not (tmp_path / "tampered-result" / "rows.json").exists()


def test_concurrent_materializations_are_isolated_and_same_id_has_one_winner(
    tmp_path: Path,
) -> None:
    materializer = GroundedResultStreamMaterializer(tmp_path)

    def write_distinct(index: int) -> GroundedResultArtifactReceipt:
        return materializer.materialize_batches(
            [[{"writer": index, "position": position} for position in range(20)]],
            artifact_id="result-%d" % index,
            limits=limits(max_rows=20),
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(write_distinct, range(24)))

    assert len(receipts) == 24
    assert len({receipt.rows_canonical_sha256 for receipt in receipts}) == 24
    for index, receipt in enumerate(receipts):
        rows = json.loads((tmp_path / receipt.rows_relative_path).read_text(encoding="utf-8"))
        assert {row["writer"] for row in rows} == {index}

    def compete(index: int) -> GroundedResultArtifactReceipt | GroundedResultStreamingError:
        try:
            return materializer.materialize_batches(
                [[{"writer": index}]],
                artifact_id="one-winner",
                limits=limits(),
            )
        except GroundedResultStreamingError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(compete, range(16)))

    winners = [outcome for outcome in outcomes if isinstance(outcome, GroundedResultArtifactReceipt)]
    failures = [outcome for outcome in outcomes if isinstance(outcome, GroundedResultStreamingError)]
    assert len(winners) == 1
    assert len(failures) == 15
    assert all(error.code == GroundedResultStreamCode.ARTIFACT_CONFLICT for error in failures)


class FakeServerCursor:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = list(rows)
        self.execute_calls: list[tuple[str, tuple[Any, ...] | None]] = []
        self.fetch_sizes: list[int] = []
        self.closed = False

    def execute(self, sql: str, params: tuple[Any, ...] | None = None) -> None:
        self.execute_calls.append((sql, params))

    def fetchmany(self, size: int) -> list[dict[str, Any]]:
        self.fetch_sizes.append(size)
        batch = self.rows[:size]
        del self.rows[:size]
        return batch

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeServerCursor) -> None:
        self.server_cursor = cursor
        self.cursor_classes: list[type[Any]] = []
        self.closed = False

    def cursor(self, cursor_class: type[Any]) -> FakeServerCursor:
        self.cursor_classes.append(cursor_class)
        return self.server_cursor

    def close(self) -> None:
        self.closed = True


def install_server_cursor_module(monkeypatch: pytest.MonkeyPatch) -> type[Any]:
    cursor_module = types.ModuleType("pymysql.cursors")
    server_cursor_class = type("SSDictCursor", (), {})
    cursor_module.SSDictCursor = server_cursor_class
    monkeypatch.setitem(sys.modules, "pymysql.cursors", cursor_module)
    return server_cursor_class


def bind_connection(client: DatabaseClient, connection: FakeConnection) -> None:
    @contextmanager
    def opened_connection() -> Iterator[FakeConnection]:
        try:
            yield connection
        finally:
            connection.close()

    client.connection = opened_connection  # type: ignore[method-assign]


def test_database_client_uses_server_cursor_and_fetchmany(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server_cursor_class = install_server_cursor_module(monkeypatch)
    cursor = FakeServerCursor([{"position": position} for position in range(7)])
    connection = FakeConnection(cursor)
    client = DatabaseClient("jdbc:mysql://localhost:9030/db", "user", "password")
    bind_connection(client, connection)

    batches = list(
        client.stream_query_batches(
            "SELECT value FROM source",
            ["binding"],
            batch_size=3,
            timeout_seconds=5,
        )
    )

    assert [len(batch) for batch in batches] == [3, 3, 1]
    assert connection.cursor_classes == [server_cursor_class]
    assert cursor.execute_calls == [("SELECT value FROM source", ("binding",))]
    assert cursor.fetch_sizes == [3, 3, 3, 3]
    assert cursor.closed is True
    assert connection.closed is True


def test_database_stream_has_one_normal_connection_close_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_server_cursor_module(monkeypatch)
    cursor = FakeServerCursor([{"value": 1}])

    class StrictCloseConnection(FakeConnection):
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls > 1:
                raise RuntimeError("Already closed")
            super().close()

    connection = StrictCloseConnection(cursor)
    client = DatabaseClient(
        "jdbc:mysql://localhost:9030/db",
        "user",
        "password",
    )
    bind_connection(client, connection)

    assert list(
        client.stream_query_batches(
            "SELECT value FROM source",
            batch_size=10,
            timeout_seconds=5,
        )
    ) == [[{"value": 1}]]
    assert connection.close_calls == 1


def test_database_stream_cancellation_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_server_cursor_module(monkeypatch)
    cancel = threading.Event()
    fetch_started = threading.Event()

    class BlockingCursor(FakeServerCursor):
        def __init__(self) -> None:
            super().__init__([])
            self.connection: FakeConnection | None = None

        def fetchmany(self, size: int) -> list[dict[str, Any]]:
            fetch_started.set()
            assert self.connection is not None
            while not self.connection.closed:
                threading.Event().wait(0.01)
            raise OSError("connection closed")

    cursor = BlockingCursor()
    connection = FakeConnection(cursor)
    cursor.connection = connection
    client = DatabaseClient("jdbc:mysql://localhost:9030/db", "user", "password")
    bind_connection(client, connection)
    stream = client.stream_query_batches(
        "SELECT value FROM source",
        batch_size=2,
        cancel_events=[cancel],
        timeout_seconds=5,
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        outcome = pool.submit(next, stream)
        assert fetch_started.wait(1)
        cancel.set()
        with pytest.raises(DatabaseStreamError) as captured:
            outcome.result(timeout=2)

    assert captured.value.code == DatabaseStreamError.CANCELLED
    assert cursor.closed is True
    assert connection.closed is True


def test_database_stream_timeout_is_typed_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_server_cursor_module(monkeypatch)
    cursor = FakeServerCursor([])
    connection = FakeConnection(cursor)
    client = DatabaseClient("jdbc:mysql://localhost:9030/db", "user", "password")
    bind_connection(client, connection)
    observed_times = iter((0.0, 2.0))
    monkeypatch.setattr(
        repositories.time,
        "monotonic",
        lambda: next(observed_times, 2.0),
    )

    with pytest.raises(DatabaseStreamError) as captured:
        next(
            client.stream_query_batches(
                "SELECT value FROM source",
                batch_size=2,
                timeout_seconds=1,
            )
        )

    assert captured.value.code == DatabaseStreamError.TIMEOUT
    assert connection.closed is True


def test_query_adapter_maps_database_timeout_to_inactive_result_failure(
    tmp_path: Path,
) -> None:
    class TimeoutSource:
        def stream_query_batches(
            self,
            sql: str,
            params: Any = None,
            *,
            batch_size: int,
            cancel_events: Any = None,
            timeout_seconds: int | None = None,
            data_snapshot_contract: Any = None,
        ) -> Iterator[list[dict[str, Any]]]:
            del sql, params, batch_size, cancel_events, timeout_seconds, data_snapshot_contract
            yield [{"position": 1}]
            raise DatabaseStreamError(DatabaseStreamError.TIMEOUT)

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_query(
            TimeoutSource(),
            sql="SELECT value FROM source",
            artifact_id="timeout-result",
            limits=limits(),
        )

    assert_inactive_failure(captured.value, GroundedResultStreamCode.TIMEOUT)
    assert captured.value.partial.row_count == 1
    assert not (tmp_path / "timeout-result" / "rows.json").exists()


def test_query_adapter_types_source_failure_before_iterator_creation(
    tmp_path: Path,
) -> None:
    class FailedSource:
        def stream_query_batches(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise ConnectionError("source unavailable")

    with pytest.raises(GroundedResultStreamingError) as captured:
        GroundedResultStreamMaterializer(tmp_path).materialize_query(
            FailedSource(),  # type: ignore[arg-type]
            sql="SELECT value FROM source",
            artifact_id="early-failure",
            limits=limits(),
        )

    assert_inactive_failure(
        captured.value,
        GroundedResultStreamCode.SOURCE_FAILED,
    )
    assert captured.value.partial.row_count == 0
    assert list(tmp_path.iterdir()) == []


class StreamingEpochAwareDb:
    def __init__(self, epoch: str = "epoch-1") -> None:
        self.epoch = epoch
        self.stream_calls = 0

    def query(self, sql: str, params: Any = None) -> list[dict[str, Any]]:
        del sql, params
        return [{"data_epoch": self.epoch}]

    def stream_query_batches(
        self,
        sql: str,
        params: Any = None,
        *,
        batch_size: int,
        cancel_events: Any = None,
        timeout_seconds: int | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        del sql, params, batch_size, cancel_events, timeout_seconds
        self.stream_calls += 1
        yield [{"value": 1}]


def snapshot_settings(tmp_path: Path) -> Any:
    return get_settings().model_copy(
        update={
            "harness_workspace_path": str(tmp_path),
            "cache_enabled": True,
            "cache_doris_select_ttl_seconds": 60,
            "doris_datasource_environment": "test",
            "doris_data_epoch_sql": "SELECT generation AS data_epoch FROM governed_epoch",
            "doris_data_epoch_column": "data_epoch",
            "doris_cache_generation": "cache-schema-v1",
        }
    )


def test_repository_revalidates_snapshot_before_opening_stream(tmp_path: Path) -> None:
    repository = DorisRepository(snapshot_settings(tmp_path))
    database = StreamingEpochAwareDb()
    repository.db = database  # type: ignore[assignment]
    snapshot = repository.capture_data_snapshot("semantic-v1")
    database.epoch = "epoch-2"

    with pytest.raises(RuntimeError) as captured:
        list(
            repository.stream_query_batches(
                "SELECT value FROM source",
                batch_size=10,
                data_snapshot_contract=snapshot,
            )
        )

    assert "DATA_SNAPSHOT_STALE" in str(captured.value)
    assert database.stream_calls == 0


def test_repository_preserves_unsupported_snapshot_identity_while_streaming(
    tmp_path: Path,
) -> None:
    repository = DorisRepository(snapshot_settings(tmp_path))
    database = StreamingEpochAwareDb()
    repository.db = database  # type: ignore[assignment]
    snapshot = DataSnapshotContract(
        datasource_fingerprint="observed-datasource",
        datasource_environment="test",
        semantic_activation_fingerprint="semantic-v1",
        cache_generation="cache-schema-v1",
        unsupported_reason="OBSERVED_EPOCH_UNAVAILABLE",
    )

    rows = list(
        repository.stream_query_batches(
            "SELECT value FROM source",
            batch_size=10,
            data_snapshot_contract=snapshot,
        )
    )

    assert rows == [[{"value": 1}]]
    assert repository.last_data_snapshot == snapshot
    assert repository.last_cache_key == ""
    assert repository.last_cache_hit is False
