from __future__ import annotations

import json
import logging
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from merchant_ai.config import Settings, jdbc_to_pymysql_kwargs
from merchant_ai.models import MerchantInfo, PendingAnswer
from merchant_ai.services.cache import build_ttl_cache, stable_cache_key


logger = logging.getLogger(__name__)


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
            conn.close()

    def query(self, sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
        with self.connection() as conn:
            with conn.cursor() as cursor:
                if params:
                    cursor.execute(sql, tuple(params))
                else:
                    cursor.execute(sql)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

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
        self.last_degraded_reason: Dict[str, Any] = {}

    def query(self, sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
        self.last_cache_hit = False
        self.last_cache_key = ""
        params_list = list(params or [])
        cacheable = self._cacheable_query(sql)
        cache_key = stable_cache_key("doris", {"sql": normalize_sql_for_cache(sql), "params": params_list}) if cacheable else ""
        if cache_key:
            cached = self.query_cache.get(cache_key)
            if cached is not None:
                self.last_cache_hit = True
                self.last_cache_key = cache_key
                return cached
        try:
            rows = self.db.query(sql, params_list or None)
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

    def show_full_columns(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW FULL COLUMNS FROM `%s`" % safe_table)

    def show_create_table(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW CREATE TABLE `%s`" % safe_table)

    def sample_rows(self, table_name: str, merchant_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SELECT * FROM `%s` LIMIT %s" % (safe_table, max(1, min(limit, 100))))

    def profile_enum_candidates(
        self,
        table_name: str,
        merchant_id: str,
        columns: List[str],
        limit: int = 20,
    ) -> Dict[str, Any]:
        """Discover low-cardinality values from a bounded Doris table sample.

        Business meanings are deliberately not inferred here; discovered values stay
        UNREVIEWED until the semantic asset is enriched from governed business knowledge.
        """
        safe_table = safe_identifier(table_name)
        value_limit = max(1, min(int(limit or 20), 100))
        result: Dict[str, Any] = {}
        for raw_column in list(columns or [])[:24]:
            safe_column = safe_identifier(str(raw_column))
            sample_source = "`%s` TABLESAMPLE(10 PERCENT)" % safe_table
            stats = self.query_one(
                "SELECT COUNT(`%s`) AS scanned_rows, COUNT(DISTINCT `%s`) AS distinct_count FROM %s"
                % (safe_column, safe_column, sample_source)
            )
            distinct_count = int(stats.get("distinct_count") or 0)
            scanned_rows = int(stats.get("scanned_rows") or 0)
            if distinct_count <= 0 or distinct_count > value_limit:
                continue
            rows = self.query(
                "SELECT `%s` AS enum_value, COUNT(*) AS value_count FROM %s "
                "WHERE `%s` IS NOT NULL GROUP BY `%s` ORDER BY value_count DESC LIMIT %s"
                % (safe_column, sample_source, safe_column, safe_column, value_limit)
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
    return re.sub(r"\s+", " ", str(sql or "").strip())


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
    def __init__(self):
        self._answers: Dict[str, PendingAnswer] = {}

    def put(self, answer: PendingAnswer) -> None:
        self._answers[answer.id] = answer

    def get(self, answer_id: str) -> Optional[PendingAnswer]:
        return self._answers.get(answer_id)

    def remove(self, answer_id: str) -> None:
        self._answers.pop(answer_id, None)


class MerchantService:
    def __init__(self, settings: Settings, doris_repository: DorisRepository):
        self.settings = settings
        self.doris_repository = doris_repository
        self.last_degraded_reason: Dict[str, Any] = {}

    def current_merchant(self, merchant_id: str) -> MerchantInfo:
        target = merchant_id or self.settings.merchant_id
        try:
            rows = self.doris_repository.query(
                "SELECT * FROM dim_merchant_df WHERE merchant_id = %s LIMIT 1",
                [target],
            )
            if rows:
                row = rows[0]
                return MerchantInfo(
                    merchant_id=str(row.get("merchant_id") or target),
                    merchant_name=str(row.get("merchant_name") or row.get("company_name") or "yshopping商家%s" % target),
                    company_name=str(row.get("company_name") or ""),
                    rows=row,
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
