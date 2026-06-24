from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_backend"))

from merchant_ai.services.recall_index import build_index_manifest, load_index_manifest  # noqa: E402


def semantic_root() -> Path:
    return ROOT / "python_backend" / "resources" / "runtime" / "topics"


def default_manifest_path() -> Path:
    return ROOT / "python_backend" / ".merchant-ai" / "recall_index_manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild local recall index manifest for semantic assets.")
    parser.add_argument("--changed-only", action="store_true", help="Only list docs whose source hash changed.")
    parser.add_argument("--root", type=Path, default=semantic_root(), help="Semantic topics root.")
    parser.add_argument("--output", type=Path, default=default_manifest_path(), help="Manifest output path.")
    args = parser.parse_args()

    previous = load_index_manifest(args.output)
    manifest = build_index_manifest(args.root, previous, args.changed_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "indexVersion=%s docCount=%d updatedRefs=%d output=%s"
        % (manifest["indexVersion"], manifest["docCount"], len(manifest["updatedRefs"]), args.output)
    )


if __name__ == "__main__":
    main()
