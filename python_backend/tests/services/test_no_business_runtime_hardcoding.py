import ast
import json
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
    published_identifiers = published_business_identifiers()
    violations: list[tuple[str, str]] = []

    for path in RUNTIME_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(
                node.value,
                str,
            ):
                continue
            if node.value in published_identifiers:
                violations.append(
                    (
                        str(path.relative_to(BACKEND_ROOT)),
                        node.value,
                    )
                )

    assert violations == []
