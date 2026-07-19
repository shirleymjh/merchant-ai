import ast
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
TOPIC_ROOT = REPOSITORY_ROOT / "python_backend" / "resources" / "runtime" / "topics"
RUNTIME_SOURCE_ROOTS = (
    REPOSITORY_ROOT / "python_backend" / "app",
    REPOSITORY_ROOT / "python_backend" / "merchant_ai",
)
SEMANTIC_IDENTIFIER_KEYS = {"metricKey", "tableName", "topic"}


def _collect_semantic_identifiers(value: Any) -> set[str]:
    identifiers: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SEMANTIC_IDENTIFIER_KEYS and isinstance(item, str) and item.strip():
                identifiers.add(item.strip())
            identifiers.update(_collect_semantic_identifiers(item))
    elif isinstance(value, list):
        for item in value:
            identifiers.update(_collect_semantic_identifiers(item))
    return identifiers


def _published_semantic_identifiers() -> set[str]:
    identifiers = {path.name for path in TOPIC_ROOT.iterdir() if path.is_dir()}
    for path in TOPIC_ROOT.rglob("*.json"):
        try:
            identifiers.update(
                _collect_semantic_identifiers(json.loads(path.read_text(encoding="utf-8")))
            )
        except (OSError, ValueError):
            continue
    return {item for item in identifiers if item}


def test_runtime_code_does_not_embed_published_business_identifiers() -> None:
    governed = _published_semantic_identifiers()
    failures: list[str] = []
    for root in RUNTIME_SOURCE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and node.value in governed
                ):
                    failures.append(
                        f"{path.relative_to(REPOSITORY_ROOT)}:{node.lineno}: {node.value}"
                    )

    assert not failures, (
        "Published Topic, table, and metric identifiers belong in semantic assets, "
        "not runtime source:\n" + "\n".join(failures)
    )
