from __future__ import annotations

import hashlib
import json
import logging
import inspect
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any, Dict, Iterable, Iterator, List, Optional

from merchant_ai.config import Settings, jdbc_to_pymysql_kwargs
from merchant_ai.models import DataSnapshotContract, MerchantInfo, PendingAnswer
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key


logger = logging.getLogger(__name__)


class DatabaseStreamError(RuntimeError):
    CANCELLED = "DATABASE_STREAM_CANCELLED"
    TIMEOUT = "DATABASE_STREAM_TIMEOUT"
    FAILED = "DATABASE_STREAM_FAILED"

    def __init__(self, code: str, *, cause_type: str = "") -> None:
        self.code = str(code or self.FAILED)
        self.cause_type = str(cause_type or "")
        message = self.code
        if self.cause_type:
            message = "%s:%s" % (message, self.cause_type)
        super().__init__(message)


def degraded_reason(component: str, operation: str, exc: Exception) -> Dict[str, Any]:
    return {
        "component": component,
        "operation": operation,
        "errorType": exc.__class__.__name__,
        "message": str(exc)[:500],
        "timestamp": datetime.now().isoformat(),
    }


def log_degraded(component: str, operation: str, exc: Exception) -> Dict[str, Any]:
    reason = degraded_reason(component, operation, exc)
    logger.warning(
        "%s degraded during %s: %s: %s",
        component,
        operation,
        reason["errorType"],
        reason["message"],
    )
    return reason


class DatabaseClient:
    def __init__(self, jdbc_url: str, username: str, password: str, read_timeout_seconds: int = 30):
        self.kwargs = jdbc_to_pymysql_kwargs(jdbc_url, username, password)
        self.read_timeout_seconds = max(1, int(read_timeout_seconds or 30))
        self.available = True
        self.last_degraded_reason: Dict[str, Any] = {}

    @contextmanager
    def connection(self):
        try:
            import pymysql
            from pymysql.cursors import DictCursor

            kwargs = dict(self.kwargs)
            kwargs.pop("cursorclass_name", None)
            kwargs["cursorclass"] = DictCursor
            kwargs.setdefault("connect_timeout", 5)
            kwargs.setdefault("read_timeout", self.read_timeout_seconds)
            kwargs.setdefault("write_timeout", self.read_timeout_seconds)
            conn = pymysql.connect(**kwargs)
        except Exception as exc:
            self.available = False
            self.last_degraded_reason = log_degraded("database", "connect", exc)
            raise RuntimeError("数据库连接不可用") from exc
        try:
            yield conn
        finally:
            # PyMySQL raises ``Error('Already closed')`` on a second close.
            # Cancellation monitors may close the socket before this owner
            # exits, so connection cleanup must be idempotent.
            try:
                if bool(getattr(conn, "open", True)):
                    conn.close()
            except Exception as exc:
                if "already closed" not in str(exc).lower():
                    logger.warning(
                        "database connection close failed: %s: %s",
                        type(exc).__name__,
                        str(exc)[:300],
                    )

    def query(
        self,
        sql: str,
        params: Optional[Iterable[Any]] = None,
        cancel_events: Optional[Iterable[Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            stop_monitor = Event()
            cancellation_events = [event for event in (cancel_events or []) if event is not None]
            timeout = max(1, int(timeout_seconds or self.read_timeout_seconds or 1))
            deadline = time.monotonic() + timeout

            def monitor_query() -> None:
                while not stop_monitor.wait(0.05):
                    canceled = any(bool(getattr(event, "is_set", lambda: False)()) for event in cancellation_events)
                    if not canceled and time.monotonic() < deadline:
                        continue
                    try:
                        conn.close()
                    except Exception:
                        pass
                    return

            monitor = Thread(target=monitor_query, name="doris-query-cancel", daemon=True)
            monitor.start()
            try:
                if any(bool(getattr(event, "is_set", lambda: False)()) for event in cancellation_events):
                    raise RuntimeError("database query canceled")
                with conn.cursor() as cursor:
                    if params:
                        cursor.execute(sql, tuple(params))
                    else:
                        cursor.execute(sql)
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
            finally:
                stop_monitor.set()

    def stream_query_batches(
        self,
        sql: str,
        params: Optional[Iterable[Any]] = None,
        *,
        batch_size: int,
        cancel_events: Optional[Iterable[Any]] = None,
        timeout_seconds: Optional[int] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Yield bounded batches from a server-side PyMySQL cursor.

        The connection remains owned by this iterator and is closed on EOF,
        cancellation, timeout, source error, or consumer ``close()``. No
        ``fetchall`` path is used, so result size does not determine process
        memory usage.
        """

        size = max(1, int(batch_size or 1))
        cancellation_events = tuple(
            event for event in (cancel_events or ()) if event is not None
        )
        timeout = max(
            1,
            int(timeout_seconds or self.read_timeout_seconds or 1),
        )
        deadline = time.monotonic() + timeout
        with self.connection() as conn:
            try:
                from pymysql.cursors import SSDictCursor
            except Exception as exc:
                raise DatabaseStreamError(
                    DatabaseStreamError.FAILED,
                    cause_type=type(exc).__name__,
                ) from exc

            stop_monitor = Event()
            interruption: Dict[str, str] = {"code": ""}

            def cancellation_requested() -> bool:
                return any(
                    bool(getattr(event, "is_set", lambda: False)())
                    for event in cancellation_events
                )

            def interrupt_if_required() -> None:
                code = str(interruption.get("code") or "")
                if not code and cancellation_requested():
                    code = DatabaseStreamError.CANCELLED
                    interruption["code"] = code
                if not code and time.monotonic() >= deadline:
                    code = DatabaseStreamError.TIMEOUT
                    interruption["code"] = code
                if code:
                    raise DatabaseStreamError(code)

            def monitor_query() -> None:
                while not stop_monitor.wait(0.05):
                    code = ""
                    if cancellation_requested():
                        code = DatabaseStreamError.CANCELLED
                    elif time.monotonic() >= deadline:
                        code = DatabaseStreamError.TIMEOUT
                    if not code:
                        continue
                    interruption["code"] = code
                    try:
                        conn.close()
                    except Exception:
                        pass
                    return

            monitor = Thread(
                target=monitor_query,
                name="doris-stream-query-cancel",
                daemon=True,
            )
            monitor.start()
            cursor = None
            try:
                interrupt_if_required()
                cursor = conn.cursor(SSDictCursor)
                if params:
                    cursor.execute(sql, tuple(params))
                else:
                    cursor.execute(sql)
                while True:
                    interrupt_if_required()
                    rows = cursor.fetchmany(size)
                    interrupt_if_required()
                    if not rows:
                        break
                    yield [dict(row) for row in rows]
            except DatabaseStreamError:
                raise
            except Exception as exc:
                code = str(interruption.get("code") or "")
                if not code and cancellation_requested():
                    code = DatabaseStreamError.CANCELLED
                if not code and time.monotonic() >= deadline:
                    code = DatabaseStreamError.TIMEOUT
                raise DatabaseStreamError(
                    code or DatabaseStreamError.FAILED,
                    cause_type=type(exc).__name__,
                ) from exc
            finally:
                stop_monitor.set()
                if cursor is not None:
                    try:
                        cursor.close()
                    except Exception:
                        pass
                # The connection context manager is the sole normal owner of
                # connection shutdown.  The monitor may still close it early
                # for timeout/cancellation, which that owner handles
                # idempotently.
                monitor.join(timeout=0.2)

    def execute(self, sql: str, params: Optional[Iterable[Any]] = None) -> int:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                if params:
                    count = cursor.execute(sql, tuple(params))
                else:
                    count = cursor.execute(sql)
            conn.commit()
            return count


class DorisRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = DatabaseClient(settings.doris_jdbc_url, settings.doris_username, settings.doris_password, settings.doris_read_timeout_seconds)
        self.query_cache = build_ttl_cache("doris_select", settings, settings.cache_doris_select_ttl_seconds)
        self.last_cache_hit = False
        self.last_cache_key = ""
        self.last_data_snapshot = DataSnapshotContract()
        self.last_degraded_reason: Dict[str, Any] = {}

    def query(
        self,
        sql: str,
        params: Optional[Iterable[Any]] = None,
        cancel_events: Optional[Iterable[Any]] = None,
        timeout_seconds: Optional[int] = None,
        data_snapshot_contract: Optional[DataSnapshotContract] = None,
    ) -> List[Dict[str, Any]]:
        self.last_cache_hit = False
        self.last_cache_key = ""
        params_list = list(params or [])
        snapshot = data_snapshot_contract or DataSnapshotContract(
            unsupported_reason="DATA_SNAPSHOT_CONTRACT_NOT_SUPPLIED"
        )
        self.last_data_snapshot = snapshot.model_copy(deep=True)
        snapshot_current = self._snapshot_is_current(snapshot)
        cacheable = self._cacheable_query(sql) and snapshot_current
        cache_key = (
            stable_cache_key(
                "doris",
                {
                    "sql": normalize_sql_for_cache(sql),
                    "params": params_list,
                    "snapshot": snapshot.cache_identity(),
                },
            )
            if cacheable
            else ""
        )
        if cache_key:
            cached = self.query_cache.get(cache_key)
            if cached is not None:
                self.last_cache_hit = True
                self.last_cache_key = cache_key
                return cached
        try:
            query_signature = inspect.signature(self.db.query)
            supported = set(query_signature.parameters)
            kwargs: Dict[str, Any] = {}
            if "cancel_events" in supported:
                kwargs["cancel_events"] = cancel_events
            if "timeout_seconds" in supported:
                kwargs["timeout_seconds"] = timeout_seconds
            rows = self.db.query(sql, params_list or None, **kwargs)
        except Exception as exc:
            self.last_degraded_reason = log_degraded("doris_repository", "query", exc)
            raise
        if cache_key:
            self.query_cache.set(cache_key, rows)
            self.last_cache_key = cache_key
        return rows

    def query_one(self, sql: str, params: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
        rows = self.query(sql, params)
        return rows[0] if rows else {}

    def stream_query_batches(
        self,
        sql: str,
        params: Optional[Iterable[Any]] = None,
        *,
        batch_size: int,
        cancel_events: Optional[Iterable[Any]] = None,
        timeout_seconds: Optional[int] = None,
        data_snapshot_contract: Optional[DataSnapshotContract] = None,
    ) -> Iterator[List[Dict[str, Any]]]:
        """Stream an uncached Doris result while retaining snapshot identity."""

        self.last_cache_hit = False
        self.last_cache_key = ""
        snapshot = data_snapshot_contract or DataSnapshotContract(
            unsupported_reason="DATA_SNAPSHOT_CONTRACT_NOT_SUPPLIED"
        )
        self.last_data_snapshot = snapshot.model_copy(deep=True)
        self._snapshot_is_current(snapshot)
        params_list = list(params or [])
        stream: Iterator[List[Dict[str, Any]]] | None = None
        try:
            stream = self.db.stream_query_batches(
                sql,
                params_list or None,
                batch_size=max(1, int(batch_size or 1)),
                cancel_events=cancel_events,
                timeout_seconds=timeout_seconds,
            )
            for batch in stream:
                yield batch
        except Exception as exc:
            self.last_degraded_reason = log_degraded(
                "doris_repository",
                "stream_query_batches",
                exc,
            )
            raise
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

    def datasource_fingerprint(self) -> str:
        connection = jdbc_to_pymysql_kwargs(
            self.settings.doris_jdbc_url,
            self.settings.doris_username,
            self.settings.doris_password,
        )
        return stable_cache_key(
            "doris_datasource",
            {
                "host": str(connection.get("host") or ""),
                "port": int(connection.get("port") or 0),
                "database": str(connection.get("database") or ""),
                "environment": str(
                    getattr(self.settings, "doris_datasource_environment", "") or ""
                ).strip(),
            },
        )

    def capture_data_snapshot(
        self,
        semantic_activation_fingerprint: str,
    ) -> DataSnapshotContract:
        """Observe a configured data generation without claiming atomic reads.

        The epoch SQL and result column are deployment capabilities, not table
        heuristics. Missing or malformed capability configuration fails closed
        for result caching and is explicitly represented as ``UNSUPPORTED``.
        """

        environment = str(
            getattr(self.settings, "doris_datasource_environment", "") or ""
        ).strip()
        epoch_sql = str(getattr(self.settings, "doris_data_epoch_sql", "") or "").strip()
        epoch_column = str(
            getattr(self.settings, "doris_data_epoch_column", "") or ""
        ).strip()
        cache_generation = str(
            getattr(self.settings, "doris_cache_generation", "") or ""
        ).strip()
        semantic_fingerprint = str(semantic_activation_fingerprint or "").strip()
        base = {
            "datasource_fingerprint": self.datasource_fingerprint(),
            "datasource_environment": environment,
            "semantic_activation_fingerprint": semantic_fingerprint,
            "cache_generation": cache_generation,
            "captured_at": datetime.now().isoformat(),
        }
        missing = [
            name
            for name, value in (
                ("DATASOURCE_ENVIRONMENT", environment),
                ("DATA_EPOCH_SQL", epoch_sql),
                ("DATA_EPOCH_COLUMN", epoch_column),
                ("SEMANTIC_ACTIVATION_FINGERPRINT", semantic_fingerprint),
                ("CACHE_GENERATION", cache_generation),
            )
            if not value
        ]
        if missing:
            return DataSnapshotContract(
                **base,
                unsupported_reason="MISSING_" + "_AND_".join(missing),
            )
        if not self._cacheable_query(epoch_sql):
            return DataSnapshotContract(
                **base,
                unsupported_reason="DATA_EPOCH_SQL_MUST_BE_SINGLE_READ_ONLY_QUERY",
            )
        try:
            rows = self.db.query(epoch_sql)
        except Exception as exc:
            self.last_degraded_reason = log_degraded(
                "doris_repository", "capture_data_snapshot", exc
            )
            return DataSnapshotContract(
                **base,
                unsupported_reason="DATA_EPOCH_QUERY_FAILED",
            )
        if len(rows) != 1 or epoch_column not in rows[0]:
            return DataSnapshotContract(
                **base,
                unsupported_reason="DATA_EPOCH_RESULT_INVALID",
            )
        data_epoch = str(rows[0].get(epoch_column) or "").strip()
        if not data_epoch:
            return DataSnapshotContract(
                **base,
                unsupported_reason="DATA_EPOCH_EMPTY",
            )
        return DataSnapshotContract(
            **base,
            data_epoch=data_epoch,
            consistency_mode="OBSERVED_EPOCH",
        )

    def _snapshot_is_current(self, snapshot: DataSnapshotContract) -> bool:
        if not snapshot.cache_identity_complete():
            return False
        if snapshot.consistency_mode == "AS_OF_READ":
            raise RuntimeError(
                "AS_OF_READ_UNSUPPORTED: generic Doris repository cannot enforce an as-of read"
            )
        if snapshot.datasource_fingerprint != self.datasource_fingerprint():
            raise RuntimeError(
                "DATA_SNAPSHOT_DATASOURCE_MISMATCH: snapshot belongs to another datasource"
            )
        current = self.capture_data_snapshot(
            snapshot.semantic_activation_fingerprint
        )
        if not current.cache_identity_complete():
            raise RuntimeError(
                "DATA_SNAPSHOT_REVALIDATION_FAILED: current data generation is unavailable"
            )
        if current.cache_identity() != snapshot.cache_identity():
            raise RuntimeError(
                "DATA_SNAPSHOT_STALE: data, semantic or cache generation changed before query"
            )
        return True

    def revalidate_data_snapshot(
        self,
        snapshot: DataSnapshotContract,
    ) -> DataSnapshotContract:
        """Recheck a shared snapshot immediately before business execution."""

        if snapshot.cache_identity_complete():
            self._snapshot_is_current(snapshot)
            return snapshot.model_copy(deep=True)
        if (
            snapshot.datasource_fingerprint
            and snapshot.datasource_fingerprint != self.datasource_fingerprint()
        ):
            raise RuntimeError(
                "DATA_SNAPSHOT_DATASOURCE_MISMATCH: snapshot belongs to another datasource"
            )
        current = self.capture_data_snapshot(
            snapshot.semantic_activation_fingerprint
        )
        fields = (
            "datasource_fingerprint",
            "datasource_environment",
            "data_epoch",
            "consistency_mode",
            "semantic_activation_fingerprint",
            "cache_generation",
        )
        if any(
            str(getattr(current, field_name, "") or "")
            != str(getattr(snapshot, field_name, "") or "")
            for field_name in fields
        ):
            raise RuntimeError(
                "DATA_SNAPSHOT_STALE: snapshot authority changed before query"
            )
        return snapshot.model_copy(deep=True)

    def show_full_columns(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW FULL COLUMNS FROM `%s`" % safe_table)

    def show_create_table(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW CREATE TABLE `%s`" % safe_table)

    def show_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW INDEX FROM `%s`" % safe_table)

    def show_partitions(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW PARTITIONS FROM `%s`" % safe_table)

    def explain_verbose(self, sql: str) -> List[Dict[str, Any]]:
        statement = str(sql or "").strip()
        if not statement:
            raise ValueError("sql is required for EXPLAIN VERBOSE")
        return self.query("EXPLAIN VERBOSE %s" % statement)

    def sample_rows(
        self,
        table_name: str,
        merchant_id: str,
        merchant_filter_column: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return a bounded sample from exactly one governed tenant scope.

        Topic Builder is an administrative workflow, but its profiling queries
        still touch production data.  Requiring the semantic tenant column at
        this repository boundary prevents a future caller from accidentally
        turning a merchant-scoped sample into a table-wide sample.
        """

        safe_table = safe_identifier(table_name)
        safe_merchant_column = safe_identifier(merchant_filter_column)
        scoped_merchant_id = str(merchant_id or "").strip()
        if not scoped_merchant_id:
            raise ValueError("merchant_id is required for tenant-scoped sampling")
        return self.query(
            "SELECT * FROM `%s` WHERE `%s` = %%s LIMIT %s"
            % (safe_table, safe_merchant_column, max(1, min(limit, 100))),
            [scoped_merchant_id],
        )

    def profile_enum_candidates(
        self,
        table_name: str,
        merchant_id: str,
        merchant_filter_column: str,
        columns: List[str],
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Discover low-cardinality values from a bounded Doris table sample.

        Business meanings are deliberately not inferred here; discovered values stay
        UNREVIEWED until the semantic asset is enriched from governed business knowledge.
        """
        safe_table = safe_identifier(table_name)
        safe_merchant_column = safe_identifier(merchant_filter_column)
        scoped_merchant_id = str(merchant_id or "").strip()
        if not scoped_merchant_id:
            raise ValueError("merchant_id is required for tenant-scoped enum profiling")
        value_limit = max(1, min(int(limit or 20), 100))
        result: Dict[str, Any] = {}
        for raw_column in list(columns or [])[:24]:
            safe_column = safe_identifier(str(raw_column))
            if safe_column == safe_merchant_column:
                # Tenant identifiers are access-control inputs, not business
                # enums, and must never be surfaced as discovered values.
                continue
            sample_source = "`%s` TABLESAMPLE(10 PERCENT)" % safe_table
            stats = self.query_one(
                "SELECT COUNT(`%s`) AS scanned_rows, COUNT(DISTINCT `%s`) AS distinct_count FROM %s "
                "WHERE `%s` = %%s"
                % (safe_column, safe_column, sample_source, safe_merchant_column),
                [scoped_merchant_id],
            )
            distinct_count = int(stats.get("distinct_count") or 0)
            scanned_rows = int(stats.get("scanned_rows") or 0)
            if distinct_count <= 0 or distinct_count > value_limit:
                continue
            rows = self.query(
                "SELECT `%s` AS enum_value, COUNT(*) AS value_count FROM %s "
                "WHERE `%s` = %%s AND `%s` IS NOT NULL GROUP BY `%s` "
                "ORDER BY value_count DESC LIMIT %s"
                % (
                    safe_column,
                    sample_source,
                    safe_merchant_column,
                    safe_column,
                    safe_column,
                    value_limit,
                ),
                [scoped_merchant_id],
            )
            result[str(raw_column)] = {
                "values": [row.get("enum_value") for row in rows if row.get("enum_value") is not None],
                "counts": {str(row.get("enum_value")): int(row.get("value_count") or 0) for row in rows if row.get("enum_value") is not None},
                "scannedRows": scanned_rows,
                "distinctCount": distinct_count,
                "coverage": 0.1,
                "exhaustive": False,
                "reviewStatus": "UNREVIEWED",
            }
        return result

    def clear_cache(self) -> None:
        self.query_cache.clear()

    def cache_trace(self) -> Dict[str, Any]:
        trace = self.query_cache.trace()
        trace["lastCacheHit"] = self.last_cache_hit
        trace["lastCacheKey"] = self.last_cache_key
        trace["lastDataSnapshot"] = self.last_data_snapshot.model_dump(
            by_alias=True,
            exclude={"captured_at"},
        )
        if self.last_degraded_reason:
            trace["lastDegradedReason"] = self.last_degraded_reason
        return trace

    def _cacheable_query(self, sql: str) -> bool:
        text = str(sql or "").strip().lower()
        if not text:
            return False
        if not (text.startswith("select") or text.startswith("show ")):
            return False
        if ";" in text.rstrip(";"):
            return False
        volatile = [" rand(", " random(", " now(", " current_timestamp", " uuid("]
        return not any(token in " " + text for token in volatile)


def normalize_sql_for_cache(sql: str) -> str:
    return " ".join(str(sql or "").split())


class AnswerRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = DatabaseClient(settings.answer_jdbc_url, settings.answer_username, settings.answer_password, settings.doris_read_timeout_seconds)
        self.available = True
        self.last_degraded_reason: Dict[str, Any] = {}
        self.init_schema()

    def init_schema(self) -> None:
        sql_path = self.settings.resolved_sql_path / "merchant_ai_answer.sql"
        try:
            if sql_path.exists():
                self.db.execute(sql_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.available = False
            self.last_degraded_reason = log_degraded("answer_repository", "init_schema", exc)

    def exists(self, answer_id: str) -> bool:
        if not self.available:
            return False
        try:
            rows = self.db.query("SELECT COUNT(*) AS cnt FROM merchant_ai_answer WHERE id = %s", [answer_id])
            return bool(rows and int(rows[0].get("cnt") or 0) > 0)
        except Exception as exc:
            self.last_degraded_reason = log_degraded("answer_repository", "exists", exc)
            return False

    def insert_answer(self, pending: PendingAnswer, adopted: bool = False, liked: bool = False, disliked: bool = False) -> bool:
        if not self.available:
            return False
        try:
            if self.exists(pending.id):
                self.db.execute(
                    """
                    UPDATE merchant_ai_answer
                    SET question=%s, answer=%s, is_adopted=%s, like_flag=%s, dislike_flag=%s,
                        merchant_id=%s, merchant_name=%s, question_category_name=%s, doris_tables=%s,
                        suggested_questions=%s, langfuse_trace_id=%s, langfuse_session_id=%s, modify_time=%s
                    WHERE id=%s
                    """,
                    [
                        pending.question,
                        pending.answer,
                        1 if adopted else 0,
                        1 if liked else 0,
                        1 if disliked else 0,
                        pending.merchant_id,
                        pending.merchant_name,
                        pending.category_name,
                        pending.doris_tables,
                        pending.suggested_questions,
                        pending.langfuse_trace_id,
                        pending.langfuse_session_id,
                        datetime.now(),
                        pending.id,
                    ],
                )
            else:
                self.db.execute(
                    """
                    INSERT INTO merchant_ai_answer
                    (id, question, answer, is_adopted, like_flag, dislike_flag, merchant_id, merchant_name,
                     question_category_name, doris_tables, suggested_questions, langfuse_trace_id,
                     langfuse_session_id, create_time, modify_time)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        pending.id,
                        pending.question,
                        pending.answer,
                        1 if adopted else 0,
                        1 if liked else 0,
                        1 if disliked else 0,
                        pending.merchant_id,
                        pending.merchant_name,
                        pending.category_name,
                        pending.doris_tables,
                        pending.suggested_questions,
                        pending.langfuse_trace_id,
                        pending.langfuse_session_id,
                        pending.create_time,
                        datetime.now(),
                    ],
                )
            return True
        except Exception as exc:
            self.last_degraded_reason = log_degraded("answer_repository", "insert_answer", exc)
            return False

    def update_feedback(self, answer_id: str, adopted: Optional[bool], liked: Optional[bool], disliked: Optional[bool]) -> None:
        if not self.available:
            return
        try:
            self.db.execute(
                """
                UPDATE merchant_ai_answer
                SET is_adopted = COALESCE(%s, is_adopted),
                    like_flag = COALESCE(%s, like_flag),
                    dislike_flag = COALESCE(%s, dislike_flag),
                    modify_time = %s
                WHERE id = %s
                """,
                [
                    None if adopted is None else int(adopted),
                    None if liked is None else int(liked),
                    None if disliked is None else int(disliked),
                    datetime.now(),
                    answer_id,
                ],
            )
        except Exception as exc:
            self.last_degraded_reason = log_degraded("answer_repository", "update_feedback", exc)
            return

    def recent_answers(self, merchant_id: str, limit: int = 8) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            return self.db.query(
                """
                SELECT question, answer, question_category_name, doris_tables, create_time
                FROM merchant_ai_answer
                WHERE merchant_id=%s AND is_adopted=1
                ORDER BY create_time DESC
                LIMIT %s
                """,
                [merchant_id, limit],
            )
        except Exception as exc:
            self.last_degraded_reason = log_degraded("answer_repository", "recent_answers", exc)
            return []

    def recent_answers_by_category(self, merchant_id: str, category_name: str, limit: int = 200) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        try:
            return self.db.query(
                """
                SELECT question, answer, question_category_name, doris_tables, create_time
                FROM merchant_ai_answer
                WHERE merchant_id=%s AND question_category_name LIKE %s AND is_adopted=1
                ORDER BY create_time DESC
                LIMIT %s
                """,
                [merchant_id, "%%%s%%" % category_name, limit],
            )
        except Exception as exc:
            self.last_degraded_reason = log_degraded("answer_repository", "recent_answers_by_category", exc)
            return []

    def trace(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "lastDegradedReason": self.last_degraded_reason,
        }


class PendingAnswerStore:
    """Keep feedback attribution available across runs and process restarts.

    Runtime instances use a workspace-backed spool.  The optional in-memory
    mode remains useful for isolated unit tests and lightweight adapters.
    Files are authoritative when persistence is enabled so another worker can
    consume or remove a pending answer without leaving a stale local copy.
    """

    def __init__(self, settings: Optional[Settings] = None, root: Optional[Path] = None):
        self._answers: Dict[str, PendingAnswer] = {}
        self._lock = RLock()
        self._root = Path(root) if root is not None else (
            settings.resolved_workspace_path / "pending_answers"
            if settings is not None
            else None
        )
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
            try:
                self._root.chmod(0o700)
            except OSError:
                pass

    def put(self, answer: PendingAnswer) -> None:
        with self._lock:
            self._answers[answer.id] = answer
            path = self._path(answer.id)
            if path is None:
                return
            payload = answer.model_dump(by_alias=True, mode="json")
            temporary = path.with_name(".%s.%s.%s.tmp" % (path.name, os.getpid(), uuid.uuid4().hex))
            try:
                with temporary.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False, default=str)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    temporary.chmod(0o600)
                except OSError:
                    pass
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    def get(self, answer_id: str) -> Optional[PendingAnswer]:
        with self._lock:
            path = self._path(answer_id)
            if path is None:
                return self._answers.get(answer_id)
            if not path.exists():
                self._answers.pop(answer_id, None)
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                answer = PendingAnswer.model_validate(payload)
            except (OSError, ValueError, TypeError):
                return None
            if answer.id != answer_id:
                return None
            self._answers[answer_id] = answer
            return answer

    def remove(self, answer_id: str) -> None:
        with self._lock:
            self._answers.pop(answer_id, None)
            path = self._path(answer_id)
            if path is not None:
                path.unlink(missing_ok=True)

    def _path(self, answer_id: str) -> Optional[Path]:
        if self._root is None:
            return None
        digest = hashlib.sha256(str(answer_id or "").encode("utf-8")).hexdigest()
        return self._root / (digest + ".json")


class MerchantService:
    def __init__(self, settings: Settings, doris_repository: DorisRepository, profile_binding: Optional[Dict[str, Any]] = None):
        self.settings = settings
        self.doris_repository = doris_repository
        self.profile_binding = dict(profile_binding or {})
        self.last_degraded_reason: Dict[str, Any] = {}

    def current_merchant(self, merchant_id: str) -> MerchantInfo:
        target = merchant_id or self.settings.merchant_id
        try:
            table = safe_identifier(str(self.profile_binding.get("table") or ""))
            lookup_column = safe_identifier(str(self.profile_binding.get("lookupColumn") or ""))
            id_column = safe_identifier(str(self.profile_binding.get("idColumn") or lookup_column))
            display_columns = [
                safe_identifier(str(item))
                for item in self.profile_binding.get("displayColumns") or []
                if str(item)
            ]
            context_columns = [
                safe_identifier(str(item))
                for item in self.profile_binding.get("contextColumns") or []
                if str(item)
            ]
            as_of_column = (
                safe_identifier(str(self.profile_binding.get("asOfColumn") or ""))
                if self.profile_binding.get("asOfColumn")
                else ""
            )
        except ValueError:
            table = ""
            lookup_column = ""
            id_column = ""
            display_columns = []
            context_columns = []
            as_of_column = ""
        if not table or not lookup_column:
            self.last_degraded_reason = {
                "component": "merchant_service",
                "operation": "current_merchant",
                "errorType": "MISSING_SEMANTIC_RUNTIME_BINDING",
                "message": "principal profile binding is not uniquely declared by a published semantic asset",
                "timestamp": datetime.now().isoformat(),
            }
            return MerchantInfo(merchant_id=target, merchant_name="yshopping商家%s" % target)
        live_schema_provider = getattr(self.doris_repository, "show_full_columns", None)
        if callable(live_schema_provider):
            try:
                live_rows = live_schema_provider(table)
                live_columns = {
                    str(row.get("Field") or row.get("columnName") or row.get("name") or "")
                    for row in live_rows or []
                    if isinstance(row, dict)
                }
            except Exception:
                live_columns = set()
            if live_columns:
                if lookup_column not in live_columns:
                    self.last_degraded_reason = {
                        "component": "merchant_service",
                        "operation": "current_merchant",
                        "errorType": "PROFILE_LOOKUP_COLUMN_DRIFT",
                        "message": "principal profile lookup column is absent from the live schema",
                        "timestamp": datetime.now().isoformat(),
                    }
                    return MerchantInfo(merchant_id=target, merchant_name="yshopping商家%s" % target)
                id_column = id_column if id_column in live_columns else lookup_column
                display_columns = [column for column in display_columns if column in live_columns]
                context_columns = [column for column in context_columns if column in live_columns]
                as_of_column = as_of_column if as_of_column in live_columns else ""
        try:
            selected_columns: List[str] = []
            for column in [id_column, *display_columns, *context_columns]:
                if column and column not in selected_columns:
                    selected_columns.append(column)
            projection = ", ".join("`%s`" % column for column in selected_columns)
            latest_clause = " ORDER BY `%s` DESC" % as_of_column if as_of_column else ""
            rows = self.doris_repository.query(
                "SELECT %s FROM `%s` WHERE `%s` = %%s%s LIMIT 1"
                % (projection, table, lookup_column, latest_clause),
                [target],
            )
            if rows:
                row = rows[0]
                display_name = next((str(row.get(column) or "") for column in display_columns if row.get(column)), "")
                return MerchantInfo(
                    merchant_id=str(row.get(id_column) or target),
                    merchant_name=display_name or "yshopping商家%s" % target,
                    company_name=display_name,
                    # Only stable, asset-declared tags are allowed to become
                    # always-on prompt context.  Dynamic profile metrics and
                    # sensitive identity fields remain available through
                    # governed, on-demand semantic reads.
                    rows={column: row.get(column) for column in context_columns if row.get(column) is not None},
                )
        except Exception as exc:
            self.last_degraded_reason = log_degraded("merchant_service", "current_merchant", exc)
        return MerchantInfo(merchant_id=target, merchant_name="yshopping商家%s" % target)

    def trace(self) -> Dict[str, Any]:
        return {
            "lastDegradedReason": self.last_degraded_reason,
        }


def safe_identifier(identifier: str) -> str:
    if not identifier or not all(ch.isalnum() or ch == "_" for ch in identifier):
        raise ValueError("非法标识符: %s" % identifier)
    return identifier


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
