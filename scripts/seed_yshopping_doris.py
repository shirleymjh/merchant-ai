#!/usr/bin/env python3

import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path


CONTAINER = "doris-quickstart-fe-1"
MYSQL_BASE = [
    "docker",
    "exec",
    "-i",
    CONTAINER,
    "mysql",
    "-N",
    "-B",
    "-uroot",
    "-P9030",
    "-h127.0.0.1",
]
DATABASE = "yshopping"
OUTPUT_SQL = Path("scripts/generated_yshopping_seed.sql")
TABLE_ROW_COUNT = 100
PROFILE_TABLE_ROW_COUNT = 30


def run_mysql(sql: str, database: str | None = None) -> str:
    cmd = MYSQL_BASE.copy()
    if database:
        cmd.extend(["-D", database])
    cmd.extend(["-e", sql])
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def get_tables() -> list[str]:
    rows = run_mysql(f"SHOW TABLES FROM {DATABASE};")
    return [line.strip() for line in rows.splitlines() if line.strip()]


def get_columns() -> dict[str, list[dict[str, str]]]:
    sql = (
        "SELECT table_name, column_name, data_type, is_nullable, column_key, ordinal_position, "
        "COALESCE(CAST(character_maximum_length AS STRING), '') "
        "FROM information_schema.columns "
        f"WHERE table_schema='{DATABASE}' "
        "ORDER BY table_name, ordinal_position;"
    )
    rows = run_mysql(sql)
    by_table: dict[str, list[dict[str, str]]] = defaultdict(list)
    for line in rows.splitlines():
        parts = line.split("\t")
        while len(parts) < 7:
            parts.append("")
        table_name, column_name, data_type, is_nullable, column_key, ordinal_position, max_length = parts[:7]
        by_table[table_name].append(
            {
                "name": column_name,
                "type": data_type.lower(),
                "nullable": is_nullable,
                "key": column_key,
                "position": ordinal_position,
                "max_length": max_length,
            }
        )
    return by_table


def get_current_date() -> date:
    value = run_mysql("SELECT CURRENT_DATE();")
    return datetime.strptime(value.strip(), "%Y-%m-%d").date()


def sql_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def format_sql_value(value):
    if isinstance(value, (int, float)):
        return str(value)
    return f"'{sql_escape(str(value))}'"


def apply_length_limit(value, column: dict[str, str]):
    if not isinstance(value, str):
        return value
    max_length = column.get("max_length", "")
    if max_length.isdigit():
        limit = int(max_length)
        return value[:limit]
    return value


def random_time_string(pt_value: date, seed: int) -> str:
    rng = random.Random(seed)
    return f"{pt_value:%Y-%m-%d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}"


def seed_value(table: str, column: dict[str, str], row_index: int, pt_value: date):
    name = column["name"].lower()
    data_type = column["type"]
    base_number = row_index + 1

    if name == "pt":
        return pt_value.strftime("%Y-%m-%d")

    if "time" in name and data_type in {"varchar", "char", "string", "text"}:
        return random_time_string(pt_value, hash((table, name, row_index)) & 0xFFFFFFFF)

    if name in {"seller_id", "merchant_id"}:
        if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
            return 100
        return "100"

    if name == "user_id":
        return f"user_{table}_{base_number:03d}"

    if name.endswith("_id") or "_id" in name:
        if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
            return 100000 + base_number
        return f"{name}_{base_number:03d}"

    if data_type == "date":
        return pt_value.strftime("%Y-%m-%d")

    if data_type in {"bigint", "int", "integer", "smallint", "tinyint"}:
        if name.startswith("is_"):
            return base_number % 2
        if name.endswith("_cnt") or "_cnt_" in name or name.endswith("_code"):
            return (base_number % 9) + 1
        if "status" in name:
            return (base_number % 5) + 1
        if "score" in name:
            return (base_number % 5) + 1
        if "amt" in name or "price" in name or "salary" in name:
            return 1000 + base_number * 7
        return 100 + base_number

    if data_type in {"double", "float", "decimal"}:
        return round(10.5 + base_number * 1.11, 2)

    if "mobile" in name:
        return f"1380000{base_number:04d}"

    if "email" in name:
        return f"seed{base_number}@example.com"

    if "url" in name:
        return f"https://example.com/{table}/{name}/{base_number}"

    if "json" in name:
        return json.dumps({"table": table, "row": base_number, "column": name}, ensure_ascii=True)

    if "list" in name:
        return json.dumps([f"{name}_{base_number}", f"{name}_{base_number + 1}"], ensure_ascii=True)

    if "address" in name:
        return f"Address_{base_number}_Road_{pt_value:%m%d}"

    if "note" in name or "remark" in name or "desc" in name or "content" in name or "reason" in name:
        return f"{name}_{table}_{base_number}"

    if "name" in name:
        return f"{name}_{base_number}"

    if "title" in name:
        return f"{name}_{base_number}"

    if "period" in name:
        return pt_value.strftime("%Y-%m-%d")

    if name == "currency":
        return "CNY"

    if "voucher" in name:
        return f"voucher_{base_number}"

    return f"{name}_{base_number}"


def build_rows(table: str, columns: list[dict[str, str]], pt_values: list[date]) -> list[list]:
    row_count = PROFILE_TABLE_ROW_COUNT if table == "ads_merchant_profile" else TABLE_ROW_COUNT
    rows: list[list] = []
    for row_index in range(row_count):
        pt_value = pt_values[row_index % len(pt_values)]
        row = [apply_length_limit(seed_value(table, column, row_index, pt_value), column) for column in columns]
        rows.append(row)
    return rows


def alter_partitions(tables: list[str]) -> None:
    run_mysql("ADMIN SET FRONTEND CONFIG ('dynamic_partition_check_interval_seconds' = '5');", DATABASE)
    properties = (
        "'dynamic_partition.enable'='true',"
        "'dynamic_partition.time_unit'='DAY',"
        "'dynamic_partition.start'='-30',"
        "'dynamic_partition.end'='1',"
        "'dynamic_partition.prefix'='p',"
        "'dynamic_partition.create_history_partition'='true',"
        "'dynamic_partition.history_partition_num'='30'"
    )
    for table in tables:
        run_mysql(f"ALTER TABLE {DATABASE}.{table} SET ({properties});", DATABASE)


def build_sql(tables: list[str], columns_by_table: dict[str, list[dict[str, str]]], base_date: date) -> str:
    pt_values = [base_date - timedelta(days=offset) for offset in range(29, -1, -1)]
    statements = ["SET enable_insert_strict = true;"]

    for table in tables:
        columns = columns_by_table[table]
        column_names = ", ".join(f"`{column['name']}`" for column in columns)
        rows = build_rows(table, columns, pt_values)
        values_sql = []
        for row in rows:
            values_sql.append("(" + ", ".join(format_sql_value(value) for value in row) + ")")

        statements.append(f"TRUNCATE TABLE {DATABASE}.{table};")
        statements.append(
            f"INSERT INTO {DATABASE}.{table} ({column_names}) VALUES\n" + ",\n".join(values_sql) + ";"
        )

    return "\n\n".join(statements) + "\n"


def verify_counts(tables: list[str]) -> None:
    for table in tables:
        count_sql = (
            f"SELECT '{table}', COUNT(*), MIN(pt), MAX(pt) FROM {DATABASE}.{table};"
            if table != "dim_merchant_df"
            else f"SELECT '{table}', COUNT(*), MIN(pt), MAX(pt) FROM {DATABASE}.{table};"
        )
        print(run_mysql(count_sql, DATABASE))


def main() -> int:
    tables = get_tables()
    columns_by_table = get_columns()
    base_date = get_current_date()

    alter_partitions(tables)
    time.sleep(8)

    sql_text = build_sql(tables, columns_by_table, base_date)
    OUTPUT_SQL.write_text(sql_text, encoding="utf-8")

    subprocess.run(MYSQL_BASE + ["-D", DATABASE], input=sql_text, text=True, check=True)

    verify_counts(tables)
    print(f"SQL written to {OUTPUT_SQL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
