from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from merchant_ai.config import Settings, jdbc_to_pymysql_kwargs
from merchant_ai.models import MerchantInfo, PendingAnswer


class DatabaseClient:
    def __init__(self, jdbc_url: str, username: str, password: str):
        self.kwargs = jdbc_to_pymysql_kwargs(jdbc_url, username, password)
        self.available = True

    @contextmanager
    def connection(self):
        try:
            import pymysql
            from pymysql.cursors import DictCursor

            kwargs = dict(self.kwargs)
            kwargs.pop("cursorclass_name", None)
            kwargs["cursorclass"] = DictCursor
            kwargs.setdefault("connect_timeout", 5)
            kwargs.setdefault("read_timeout", 30)
            kwargs.setdefault("write_timeout", 30)
            conn = pymysql.connect(**kwargs)
        except Exception as exc:
            self.available = False
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
        self.db = DatabaseClient(settings.doris_jdbc_url, settings.doris_username, settings.doris_password)

    def query(self, sql: str, params: Optional[Iterable[Any]] = None) -> List[Dict[str, Any]]:
        return self.db.query(sql, params)

    def query_one(self, sql: str, params: Optional[Iterable[Any]] = None) -> Dict[str, Any]:
        rows = self.query(sql, params)
        return rows[0] if rows else {}

    def show_full_columns(self, table_name: str) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SHOW FULL COLUMNS FROM `%s`" % safe_table)

    def sample_rows(self, table_name: str, merchant_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        safe_table = safe_identifier(table_name)
        return self.query("SELECT * FROM `%s` LIMIT %s" % (safe_table, max(1, min(limit, 100))))


class AnswerRepository:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = DatabaseClient(settings.answer_jdbc_url, settings.answer_username, settings.answer_password)
        self.available = True
        self.init_schema()

    def init_schema(self) -> None:
        sql_path = self.settings.resolved_sql_path / "merchant_ai_answer.sql"
        try:
            if sql_path.exists():
                self.db.execute(sql_path.read_text(encoding="utf-8"))
        except Exception:
            self.available = False

    def exists(self, answer_id: str) -> bool:
        if not self.available:
            return False
        try:
            rows = self.db.query("SELECT COUNT(*) AS cnt FROM merchant_ai_answer WHERE id = %s", [answer_id])
            return bool(rows and int(rows[0].get("cnt") or 0) > 0)
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
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
        except Exception:
            return []


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
        except Exception:
            pass
        return MerchantInfo(merchant_id=target, merchant_name="yshopping商家%s" % target)


def safe_identifier(identifier: str) -> str:
    if not identifier or not all(ch.isalnum() or ch == "_" for ch in identifier):
        raise ValueError("非法标识符: %s" % identifier)
    return identifier


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
