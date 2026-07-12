from __future__ import annotations

import argparse

from merchant_ai.config import get_settings
from merchant_ai.services.distributed_workers import DistributedSubAgentWorker, builtin_worker_handlers


def main() -> None:
    parser = argparse.ArgumentParser(description="Merchant AI durable sub-agent worker")
    parser.add_argument("--once", action="store_true", help="claim and execute at most one task")
    parser.add_argument("--kinds", default="", help="comma-separated task kinds; defaults to all built-in handlers")
    parser.add_argument("--worker-id", default="", help="stable worker identity for leases and observability")
    args = parser.parse_args()

    settings = get_settings()
    handlers = builtin_worker_handlers(settings)
    worker = DistributedSubAgentWorker(settings, handlers=handlers, worker_id=args.worker_id)
    kinds = [item.strip() for item in args.kinds.split(",") if item.strip()] or list(handlers)
    if args.once:
        worker.run_once(kinds)
        return
    worker.run_forever(kinds)


if __name__ == "__main__":
    main()
