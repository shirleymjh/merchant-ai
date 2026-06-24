from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, Optional


def stable_cache_key(namespace: str, payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return "%s:%s" % (namespace, digest)


class TTLCache:
    """Small in-process TTL LRU cache for deterministic harness artifacts."""

    def __init__(self, name: str, max_entries: int = 256, ttl_seconds: int = 300):
        self.name = name
        self.max_entries = max(1, int(max_entries or 1))
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self._items: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0

    def get(self, key: str) -> Optional[Any]:
        if not key or self.ttl_seconds <= 0:
            self.misses += 1
            return None
        item = self._items.get(key)
        if item is None:
            self.misses += 1
            return None
        expires_at, value = item
        if expires_at < time.time():
            self._items.pop(key, None)
            self.evictions += 1
            self.misses += 1
            return None
        self._items.move_to_end(key)
        self.hits += 1
        return deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        if not key or self.ttl_seconds <= 0:
            return
        self._items[key] = (time.time() + self.ttl_seconds, deepcopy(value))
        self._items.move_to_end(key)
        self.sets += 1
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)
            self.evictions += 1

    def clear(self) -> None:
        self._items.clear()

    def trace(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entries": len(self._items),
            "maxEntries": self.max_entries,
            "ttlSeconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
        }

