from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_backend"))

from merchant_ai.config import get_settings  # noqa: E402
from merchant_ai.services.memory import EnterpriseMemoryStore, StructuredMemoryStore, normalize_memory  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate local JSON merchant memory into enterprise MySQL memory store.")
    parser.add_argument("--workspace", type=Path, default=None, help="Harness workspace containing memory/*.memory.json.")
    parser.add_argument("--merchant-id", default="", help="Only migrate one merchant id.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing MySQL.")
    parser.add_argument("--rebuild-vector", action="store_true", help="Rebuild ES memory vector index after MySQL write.")
    args = parser.parse_args()

    settings = get_settings()
    if args.workspace:
        settings = settings.model_copy(update={"harness_workspace_path": str(args.workspace)})
    if not settings.memory_mysql_jdbc_url and not args.dry_run:
        raise SystemExit("YSHOPPING_MEMORY_MYSQL_JDBC_URL is required unless --dry-run is used.")

    source = StructuredMemoryStore(settings)
    target = EnterpriseMemoryStore(settings.model_copy(update={"memory_backend": "mysql", "memory_index_async": False}))
    memory_dir = source.memory_path(args.merchant_id or settings.merchant_id).parent
    paths = sorted(memory_dir.glob("*.memory.json"))
    migrated = 0
    skipped = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            print("skip path=%s error=%s" % (path, str(exc)[:200]))
            skipped += 1
            continue
        merchant_id = str(payload.get("merchantId") or path.name.removesuffix(".memory.json"))
        if args.merchant_id and merchant_id != args.merchant_id:
            continue
        memory = normalize_memory(payload, merchant_id)
        print(
            "memory merchant=%s events=%d preferences=%d facts=%d conflicts=%d source=%s"
            % (
                merchant_id,
                len(memory.get("events") or []),
                len(memory.get("preferences") or []),
                len(memory.get("facts") or []),
                len(memory.get("conflicts") or []),
                path,
            )
        )
        if args.dry_run:
            migrated += 1
            continue
        saved = target.save(merchant_id, memory)
        if args.rebuild_vector:
            target.vector_index.sync_memory(saved)
        migrated += 1
    print("done migrated=%d skipped=%d memory_dir=%s" % (migrated, skipped, memory_dir))


if __name__ == "__main__":
    main()
