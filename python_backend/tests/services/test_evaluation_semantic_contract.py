import json
from pathlib import Path

from merchant_ai.config import get_settings
from merchant_ai.services.assets import TopicAssetService


GOLDEN_CASES = Path(__file__).resolve().parents[2] / "resources" / "evaluation" / "golden_cases.jsonl"


def test_golden_evaluation_refs_only_published_semantic_assets() -> None:
    assets = TopicAssetService(get_settings())
    published_tables: set[str] = set()
    published_metrics: set[str] = set()
    for topic in assets.all_topic_names():
        for item in assets.load_manifest(topic):
            table = str(item.get("tableName") or "")
            published_tables.add(table)
            published_metrics.update(
                str(metric.get("metricKey") or "")
                for metric in assets.load_table_metrics(topic, table)
                if str(metric.get("metricKey") or "")
            )

    violations: list[tuple[str, str, str]] = []
    for line in GOLDEN_CASES.read_text(encoding="utf-8").splitlines():
        case = json.loads(line)
        case_id = str(case.get("id") or "")
        violations.extend(
            (case_id, "metric", str(metric))
            for metric in case.get("expectedMetrics") or []
            if metric not in published_metrics
        )
        violations.extend(
            (case_id, "table", str(table))
            for table in case.get("expectedTables") or []
            if table not in published_tables
        )

    assert violations == []
