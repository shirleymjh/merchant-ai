import json
import re
from pathlib import Path


BACKEND_ROOT = Path(__file__).resolve().parents[2]
TOPIC_ROOT = BACKEND_ROOT / "resources" / "runtime" / "topics"
RUNTIME_ROOT = BACKEND_ROOT / "merchant_ai"


def published_business_identifiers() -> set[str]:
    identifiers: set[str] = set()
    for path in TOPIC_ROOT.glob("*/tables/*/asset.json"):
        asset = json.loads(path.read_text(encoding="utf-8"))
        table = str(asset.get("tableName") or "").strip()
        if table:
            identifiers.add(table)
        for metric in asset.get("metrics") or []:
            if not isinstance(metric, dict):
                continue
            metric_key = str(metric.get("metricKey") or "").strip()
            if metric_key:
                identifiers.add(metric_key)
    return identifiers


def test_runtime_python_does_not_embed_published_table_or_metric_literals() -> None:
    quoted_identifier_patterns = {
        identifier: re.compile(r"(['\"])" + re.escape(identifier) + r"\1")
        for identifier in published_business_identifiers()
    }
    violations: list[tuple[str, str]] = []

    for path in RUNTIME_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for identifier, pattern in quoted_identifier_patterns.items():
            if pattern.search(source):
                violations.append((str(path.relative_to(BACKEND_ROOT)), identifier))

    assert violations == []
