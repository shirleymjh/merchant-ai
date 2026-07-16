from __future__ import annotations

import json

import pytest

from merchant_ai.config import get_settings
from merchant_ai.models import (
    AnswerMode,
    NodeExecutionContext,
    PlanningAssetEntry,
    PlanningAssetPack,
    QuestionIntent,
    SqlValidationResult,
)
from merchant_ai.services.query import NodeWorkerExecutor, canonical_sql, canonical_sql_hash
from merchant_ai.services.query_contracts import sql_repair_gaps_from_task_results


INITIAL_SQL = "SELECT seller_id, name, status FROM test_table LIMIT 1"
CHANGED_SQL = "SELECT seller_id, name, status FROM test_table WHERE status IS NOT NULL LIMIT 1"
FINAL_SQL = "SELECT seller_id, name, status FROM test_table WHERE status = 'active' LIMIT 1"


class RecordingDoris:
    def __init__(self) -> None:
        self.sqls: list[str] = []

    def query(self, sql, params=None):
        self.sqls.append(sql)
        return [{"seller_id": "100", "name": "店铺", "status": "active"}]


class SequenceValidator:
    def __init__(self, results: list[SqlValidationResult]) -> None:
        self.results = results
        self.sqls: list[str] = []

    def validate(self, sql, asset_pack):
        self.sqls.append(sql)
        index = min(len(self.sqls) - 1, len(self.results) - 1)
        return self.results[index].model_copy(deep=True)


class SameSqlRepairLlm:
    configured = True
    last_error = ""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.payloads.append(json.loads(user_prompt))
        return {"sql": INITIAL_SQL}


class EquivalentFormattingRepairLlm:
    configured = True
    last_error = ""

    def __init__(self) -> None:
        self.payloads: list[dict] = []

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.payloads.append(json.loads(user_prompt))
        return {
            "sql": "  select `seller_id`,  `name`, `status`\nFROM `test_table` limit 1;  ",
        }


class ProgressingRepairLlm:
    configured = True
    last_error = ""

    def __init__(self) -> None:
        self.payloads: list[dict] = []
        self.repairs = [CHANGED_SQL, FINAL_SQL]

    def json_chat(self, system_prompt, user_prompt, fallback=None):
        self.payloads.append(json.loads(user_prompt))
        return {"sql": self.repairs[len(self.payloads) - 1]}


def public_detail_pack() -> PlanningAssetPack:
    columns = ["seller_id", "name", "status"]
    return PlanningAssetPack(
        tables=[PlanningAssetEntry(table="test_table", columns=columns)],
        fields=[
            PlanningAssetEntry(
                key=column,
                table="test_table",
                metadata={
                    "semantic": {
                        "columnName": column,
                        "visibilityPolicy": {"level": "public"},
                        "maskingPolicy": {"strategy": "none"},
                    }
                },
            )
            for column in columns
        ],
    )


def repair_intent() -> QuestionIntent:
    return QuestionIntent(
        question="查询店铺状态",
        intent_type="VALID",
        answer_mode=AnswerMode.DETAIL,
        plan_task_id="shop_status",
        preferred_table="test_table",
        sql=INITIAL_SQL,
        output_keys=["seller_id", "name", "status"],
        required_evidence=["seller_id", "name", "status"],
        days=7,
        limit=1,
    )


def invalid_result(code: str = "TEST_SQL_ERROR", message: str = "test SQL validation failed") -> SqlValidationResult:
    return SqlValidationResult(
        valid=False,
        error_code=code,
        message=message,
        base_tables=["test_table"],
    )


@pytest.mark.parametrize("llm_type", [SameSqlRepairLlm, EquivalentFormattingRepairLlm])
def test_sql_repair_stops_before_revalidating_canonical_equivalent_sql(llm_type):
    llm = llm_type()
    doris = RecordingDoris()
    validator = SequenceValidator([invalid_result()])
    settings = get_settings().model_copy(update={"agent_sql_repair_rounds": 3})
    worker = NodeWorkerExecutor(llm, doris, validator, settings)

    result = worker.execute_node(
        repair_intent(),
        public_detail_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    assert not result.success
    assert len(validator.sqls) == 1
    assert doris.sqls == []
    assert len(llm.payloads) == 1
    assert "REPAIR_NO_PROGRESS" in result.query_bundle.error
    assert len(result.sql_repairs) == 1
    attempt = result.sql_repairs[0]
    assert attempt.error_code == "REPAIR_NO_PROGRESS"
    assert attempt.source_error_code == "TEST_SQL_ERROR"
    assert attempt.status == "no_progress"
    assert not attempt.progressed
    assert attempt.exhausted
    assert attempt.input_sql_hash == attempt.output_sql_hash
    gaps = sql_repair_gaps_from_task_results([result])
    assert [gap.code for gap in gaps] == ["REPAIR_NO_PROGRESS"]
    assert gaps[0].severity == "blocking"


def test_sql_repair_passes_prior_round_observation_and_executes_only_changed_sql():
    llm = ProgressingRepairLlm()
    doris = RecordingDoris()
    validator = SequenceValidator(
        [
            invalid_result("FIRST_ERROR", "first validation observation"),
            invalid_result("SECOND_ERROR", "second validation observation"),
            SqlValidationResult(valid=True, message="passed", base_tables=["test_table"]),
        ]
    )
    settings = get_settings().model_copy(update={"agent_sql_repair_rounds": 3})
    worker = NodeWorkerExecutor(llm, doris, validator, settings)

    result = worker.execute_node(
        repair_intent(),
        public_detail_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    assert result.success
    assert len(validator.sqls) == 3
    assert [canonical_sql(sql) for sql in validator.sqls] == [
        canonical_sql(INITIAL_SQL),
        canonical_sql(CHANGED_SQL),
        canonical_sql(FINAL_SQL),
    ]
    assert len(doris.sqls) == 1
    assert canonical_sql(doris.sqls[0]) == canonical_sql(FINAL_SQL)
    assert len(llm.payloads) == 2
    assert llm.payloads[0]["previousRepairObservations"] == []
    previous = llm.payloads[1]["previousRepairObservations"]
    assert len(previous) == 1
    assert previous[0]["sourceErrorCode"] == "FIRST_ERROR"
    assert previous[0]["status"] == "progressed"
    assert previous[0]["observation"]
    assert [attempt.round for attempt in result.sql_repairs] == [1, 2]
    assert all(attempt.progressed and not attempt.exhausted for attempt in result.sql_repairs)
    assert len({attempt.output_sql_hash for attempt in result.sql_repairs}) == 2


def test_sql_repair_exhaustion_remains_typed_after_real_progress():
    llm = ProgressingRepairLlm()
    doris = RecordingDoris()
    validator = SequenceValidator(
        [
            invalid_result("FIRST_ERROR", "first validation observation"),
            invalid_result("SECOND_ERROR", "second validation observation"),
            invalid_result("THIRD_ERROR", "third validation observation"),
        ]
    )
    settings = get_settings().model_copy(update={"agent_sql_repair_rounds": 2})
    worker = NodeWorkerExecutor(llm, doris, validator, settings)

    result = worker.execute_node(
        repair_intent(),
        public_detail_pack(),
        "",
        NodeExecutionContext(merchant_id="100"),
    )

    assert not result.success
    assert len(validator.sqls) == 3
    assert doris.sqls == []
    assert "SQL_REPAIR_EXHAUSTED" in result.query_bundle.error
    assert len(result.sql_repairs) == 2
    assert result.sql_repairs[-1].progressed
    assert result.sql_repairs[-1].exhausted
    assert result.sql_repairs[-1].status == "exhausted"
    gaps = sql_repair_gaps_from_task_results([result])
    assert [gap.code for gap in gaps] == ["SQL_REPAIR_EXHAUSTED"]


def test_canonical_sql_hash_ignores_formatting_but_preserves_query_changes():
    equivalent = " select `seller_id`, `name`, `status` from `test_table` limit 1; "

    assert canonical_sql_hash(INITIAL_SQL) == canonical_sql_hash(equivalent)
    assert canonical_sql_hash(INITIAL_SQL) != canonical_sql_hash(CHANGED_SQL)
    assert canonical_sql_hash("SELECT 1; SELECT 2") != canonical_sql_hash("SELECT 1; SELECT 3")
