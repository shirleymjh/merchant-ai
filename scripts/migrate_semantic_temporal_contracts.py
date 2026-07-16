from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_backend"))

from merchant_ai.config import get_settings  # noqa: E402
from merchant_ai.services.assets import TopicAssetService, validate_semantic_asset  # noqa: E402
from merchant_ai.services.semantic_asset_migrations import migrate_published_semantic_asset  # noqa: E402


def default_semantic_root() -> Path:
    return ROOT / "python_backend" / "resources" / "runtime" / "topics"


def write_json_atomically(path: Path, payload: Dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize executable temporal contracts in published semantic assets."
    )
    parser.add_argument("--root", type=Path, default=default_semantic_root())
    parser.add_argument("--apply", action="store_true", help="Write validated changes; default is dry-run.")
    args = parser.parse_args()

    root = args.root.resolve()
    settings = get_settings().model_copy(update={"topic_path": str(root)})
    assets = TopicAssetService(settings)
    changed_assets = 0
    changed_fields = 0
    validation_errors: list[dict[str, Any]] = []
    migration_errors: list[str] = []

    candidates: list[tuple[Path, Dict[str, Any], list[str]]] = []
    for path in sorted(root.glob("*/tables/*/asset.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if str(payload.get("status") or "PUBLISHED").upper() != "PUBLISHED":
            continue
        migrated, changes, errors = migrate_published_semantic_asset(payload)
        migration_errors.extend("%s: %s" % (path, error) for error in errors)
        topic = str(migrated.get("topic") or path.parents[2].name)
        validation = validate_semantic_asset(migrated, assets.load_relationships(topic))
        for error in validation.get("errors") or []:
            validation_errors.append(
                {
                    "path": str(path),
                    "topic": topic,
                    "table": str(migrated.get("tableName") or path.parent.name),
                    **error,
                }
            )
        if changes:
            candidates.append((path, migrated, changes))
            changed_assets += 1
            changed_fields += len(changes)

    report = {
        "success": not migration_errors and not validation_errors,
        "mode": "apply" if args.apply else "dry_run",
        "root": str(root),
        "changedAssets": changed_assets,
        "changedFields": changed_fields,
        "migrationErrors": migration_errors,
        "validationErrors": validation_errors,
        "assets": [
            {"path": str(path), "changeCount": len(changes), "changes": changes}
            for path, _, changes in candidates
        ],
    }
    if migration_errors or validation_errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if args.apply:
        for path, migrated, _ in candidates:
            write_json_atomically(path, migrated)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

