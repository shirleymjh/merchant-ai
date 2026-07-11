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
            "backend": "memory",
            "entries": len(self._items),
            "maxEntries": self.max_entries,
            "ttlSeconds": self.ttl_seconds,
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "evictions": self.evictions,
        }


class RedisTTLCache:
    """Redis-backed TTL cache with in-process fallback for local development."""

    def __init__(
        self,
        name: str,
        max_entries: int = 256,
        ttl_seconds: int = 300,
        redis_url: str = "",
        namespace: str = "merchant_ai",
        socket_timeout_seconds: float = 1.0,
    ):
        self.name = name
        self.max_entries = max(1, int(max_entries or 1))
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self.namespace = _safe_namespace(namespace)
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.evictions = 0
        self.last_error = ""
        self._fallback = TTLCache(name, max_entries, ttl_seconds)
        self._client = None
        self.available = False
        if redis_url and self.ttl_seconds > 0:
            try:
                import redis

                timeout = max(0.05, float(socket_timeout_seconds or 1.0))
                self._client = redis.Redis.from_url(
                    redis_url,
                    socket_timeout=timeout,
                    socket_connect_timeout=timeout,
                    decode_responses=False,
                )
                self._client.ping()
                self.available = True
            except Exception as exc:
                self.last_error = str(exc)[:200]
                self._client = None
                self.available = False

    def get(self, key: str) -> Optional[Any]:
        if not key or self.ttl_seconds <= 0:
            self.misses += 1
            return None
        if not self.available or self._client is None:
            return self._fallback.get(key)
        try:
            raw = self._client.get(self._key(key))
            if raw is None:
                self.misses += 1
                return None
            self.hits += 1
            return deepcopy(json_cache_loads(raw))
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False
            return self._fallback.get(key)

    def set(self, key: str, value: Any) -> None:
        if not key or self.ttl_seconds <= 0:
            return
        if not self.available or self._client is None:
            self._fallback.set(key, value)
            return
        try:
            self._client.setex(self._key(key), self.ttl_seconds, json_cache_dumps(deepcopy(value)))
            self.sets += 1
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False
            self._fallback.set(key, value)

    def clear(self) -> None:
        self._fallback.clear()
        if not self.available or self._client is None:
            return
        try:
            keys = list(self._client.scan_iter(match=self._key("*"), count=200))
            if keys:
                self._client.delete(*keys)
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False

    def trace(self) -> Dict[str, Any]:
        trace = {
            "name": self.name,
            "backend": "redis" if self.available else "redis+memory_fallback",
            "available": self.available,
            "namespace": self.namespace,
            "entries": self._redis_entry_count(),
            "maxEntries": self.max_entries,
            "ttlSeconds": self.ttl_seconds,
            "hits": self.hits + self._fallback.hits,
            "misses": self.misses + self._fallback.misses,
            "sets": self.sets + self._fallback.sets,
            "evictions": self.evictions + self._fallback.evictions,
        }
        if self.last_error:
            trace["lastError"] = self.last_error
        if not self.available:
            trace["fallback"] = self._fallback.trace()
        return trace

    def _key(self, key: str) -> str:
        return "%s:ttl:%s:%s" % (self.namespace, self.name, key)

    def _redis_entry_count(self) -> int:
        if not self.available or self._client is None:
            return self._fallback.trace().get("entries", 0)
        try:
            count = 0
            for _ in self._client.scan_iter(match=self._key("*"), count=200):
                count += 1
                if count >= self.max_entries:
                    break
            return count
        except Exception as exc:
            self.last_error = str(exc)[:200]
            return 0


def build_ttl_cache(name: str, settings: Any, ttl_seconds: int) -> Any:
    max_entries = int(getattr(settings, "cache_memory_max_entries", 512) or 512)
    effective_ttl = int(ttl_seconds or 0) if bool(getattr(settings, "cache_enabled", True)) else 0
    if bool(getattr(settings, "redis_enabled", False)) and bool(getattr(settings, "redis_cache_enabled", True)):
        return RedisTTLCache(
            name,
            max_entries=max_entries,
            ttl_seconds=effective_ttl,
            redis_url=str(getattr(settings, "redis_url", "") or ""),
            namespace=str(getattr(settings, "redis_namespace", "merchant_ai") or "merchant_ai"),
            socket_timeout_seconds=float(getattr(settings, "redis_socket_timeout_seconds", 1.0) or 1.0),
        )
    return TTLCache(name, max_entries, effective_ttl)


def _safe_namespace(namespace: str) -> str:
    text = str(namespace or "merchant_ai").strip()
    return "".join(ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in text) or "merchant_ai"


def json_cache_dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, default=json_cache_default, separators=(",", ":")).encode("utf-8")


def json_cache_loads(value: Any) -> Any:
    raw = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return json.loads(raw)


def json_cache_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    if isinstance(value, set):
        return sorted(value, key=str)
    raise TypeError("unsupported cache value: %s" % type(value).__name__)
