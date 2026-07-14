from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib import request as url_request

from merchant_ai.config import Settings
from merchant_ai.services.cache import json_cache_dumps, json_cache_loads
from merchant_ai.models import (
    CircuitBreakerState,
    LoadBalancerTarget,
    RuntimeAlert,
    ToolCachePolicy,
    ToolCallExecutionResult,
    ToolCallRequest,
    ToolFailureRecord,
    ToolRecoveryAction,
    ToolRuntimeMetrics,
    ToolRuntimePolicy,
)
from merchant_ai.services.tools import ToolRegistry, canonical_tool_registry, validate_tool_result_contract


_TOOL_RUNTIME_SCOPE: ContextVar[Optional[Dict[str, str]]] = ContextVar("merchant_ai_tool_runtime_scope", default=None)
_TOOL_CANCEL_EVENT: ContextVar[Any] = ContextVar("merchant_ai_tool_cancel_event", default=None)


@contextmanager
def tool_runtime_scope(merchant_id: str = "", thread_id: str = "", run_id: str = ""):
    token = _TOOL_RUNTIME_SCOPE.set(
        {
            "merchantId": str(merchant_id or ""),
            "threadId": str(thread_id or ""),
            "runId": str(run_id or ""),
        }
    )
    try:
        yield
    finally:
        _TOOL_RUNTIME_SCOPE.reset(token)


def current_tool_runtime_scope() -> Dict[str, str]:
    return dict(_TOOL_RUNTIME_SCOPE.get() or {})


def current_tool_cancel_event() -> Any:
    return _TOOL_CANCEL_EVENT.get()


@contextmanager
def tool_cancel_scope(cancel_event: Any):
    token = _TOOL_CANCEL_EVENT.set(cancel_event)
    try:
        yield
    finally:
        _TOOL_CANCEL_EVENT.reset(token)


def scoped_rate_limit_key(
    tool_name: str,
    service_name: str = "",
    target: str = "",
    merchant_id: str = "",
    thread_id: str = "",
) -> str:
    scope = current_tool_runtime_scope()
    merchant = str(merchant_id or scope.get("merchantId") or "*")
    thread = str(thread_id or scope.get("threadId") or "*")
    run = str(scope.get("runId") or "*")
    return "tool=%s|service=%s|target=%s|merchant=%s|thread=%s|run=%s" % (
        str(tool_name or "tool"),
        str(service_name or "tool"),
        str(target or "default"),
        merchant,
        thread,
        run,
    )


def now_ms() -> int:
    return int(time.time() * 1000)


def classify_tool_error(error: Exception | str) -> str:
    text = str(error or "")
    lower = text.lower()
    if "cancelled" in lower or "canceled" in lower:
        return "CANCELED"
    if "timeout" in lower or "timed out" in lower:
        return "TIMEOUT"
    if "unknown column" in lower or "unknown field" in lower:
        return "UNKNOWN_COLUMN"
    if "mem_alloc_failed" in lower or "memory" in lower:
        return "MEM_ALLOC_FAILED"
    if "not found" in lower:
        return "NOT_FOUND"
    if "permission" in lower or "forbidden" in lower or "unauthorized" in lower:
        return "PERMISSION_DENIED"
    if "tool_contract_violation" in lower or "contract" in lower:
        return "TOOL_CONTRACT_VIOLATION"
    if "invalid" in lower or "bad argument" in lower:
        return "INVALID_ARGUMENT"
    if "provider" in lower or "llm" in lower or "model" in lower:
        return "PROVIDER_ERROR"
    return "ERROR"


def classify_timeout_type(error: Exception | str, service_name: str = "", source: str = "tool") -> str:
    text = str(error or "").lower()
    if "node" in source:
        return "node_timeout"
    if "run" in source:
        return "run_timeout"
    if "connect" in text or "connection timed out" in text or "connection timeout" in text:
        return "connect_timeout"
    if "read timed out" in text or "read timeout" in text or "socket read" in text:
        return "read_timeout"
    if service_name == "doris" and ("query timed out" in text or "read" in text):
        return "read_timeout"
    return "tool_timeout"


def normalize_tool_context(tool_name: str, args: Any, service_name: str = "", target: str = "") -> Dict[str, str]:
    safe_args = args if isinstance(args, dict) else {}
    raw_target = safe_args.get("_target") if isinstance(safe_args, dict) else {}
    target_name = target
    if not target_name and isinstance(raw_target, dict):
        target_name = str(raw_target.get("name") or raw_target.get("endpoint") or "")
    service = service_name or str(safe_args.get("_service") or safe_args.get("service") or "")
    merchant_id = str(
        safe_args.get("merchantId")
        or safe_args.get("merchant_id")
        or safe_args.get("sellerId")
        or safe_args.get("seller_id")
        or ""
    )
    thread_id = str(safe_args.get("threadId") or safe_args.get("thread_id") or "")
    params_hash = tool_params_hash(safe_args)
    service = service or "tool"
    circuit_key = "tool=%s|service=%s|target=%s|merchant=%s|thread=%s" % (
        tool_name,
        service,
        target_name or "default",
        merchant_id or "*",
        thread_id or "*",
    )
    fingerprint = "%s|params=%s" % (circuit_key, params_hash)
    return {
        "toolName": tool_name,
        "serviceName": service,
        "target": target_name or "default",
        "merchantId": merchant_id,
        "threadId": thread_id,
        "paramsHash": params_hash,
        "circuitKey": circuit_key,
        "fingerprint": fingerprint,
    }


def tool_params_hash(args: Any) -> str:
    safe_args = sanitize_tool_args(args)
    try:
        payload = json.dumps(safe_args, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        payload = str(safe_args)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tool_idempotency_key(tool_name: str, call_id: str, context: Dict[str, str]) -> str:
    parts = [
        "tool",
        str(tool_name or ""),
        "call",
        str(call_id or ""),
        "service",
        str(context.get("serviceName") or ""),
        "target",
        str(context.get("target") or ""),
        "params",
        str(context.get("paramsHash") or ""),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


def tool_contract_enforced(contract: Dict[str, Any]) -> bool:
    capability = contract.get("capability") if isinstance(contract, dict) else {}
    return bool(contract.get("enforced") or (isinstance(capability, dict) and capability.get("failClosed")))


def sanitize_tool_args(args: Any) -> Any:
    if not isinstance(args, dict):
        return args
    return {key: value for key, value in args.items() if not str(key).startswith("_")}


def recovery_action_for(tool_name: str, error_type: str, tool_kind: str = "") -> ToolRecoveryAction:
    kind = tool_kind or infer_tool_kind(tool_name)
    error = str(error_type or "ERROR")
    if error in {"UNKNOWN_COLUMN", "UNKNOWN_BASE_TABLE"}:
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="semantic_recall_or_graph_repair",
            retryable=False,
            fallback_tools=["semantic_read", "semantic_grep", "graph_repair"],
            message="字段或表不存在，请回到语义层确认口径或修正 QueryGraph。",
        )
    if error in {"INVALID_ARGUMENT", "PARSE_ERROR", "UNSAFE_SQL"}:
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="fix_arguments_or_use_structured_fallback",
            retryable=False,
            fallback_tools=["structured_sql_fallback", "ask_human"],
            message="参数或 SQL 不满足工具约束，请修正参数或使用结构化兜底。",
        )
    if error == "TOOL_CONTRACT_VIOLATION":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="repair_tool_handler_or_degrade",
            retryable=False,
            fallback_tools=["contract_critic", "answer_with_gap"],
            message="工具返回结果不满足注册契约，高风险工具已 fail closed。",
        )
    if error == "MEM_ALLOC_FAILED":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="reduce_query_scope_or_answer_with_gap",
            retryable=False,
            fallback_tools=["use_cache", "limit_rows", "answer_with_gap"],
            message="查询资源不足，请缩小时间范围、减少行数或返回部分证据。",
        )
    if error == "TIMEOUT":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="retry_if_policy_allows_or_degrade",
            retryable=kind in {"llm", "es", "semantic"},
            fallback_tools=["use_cache", "partial_answer", "ask_user_to_narrow"],
            message="工具超时，可按策略有限重试；不可恢复时使用缓存、部分结果或澄清。",
        )
    if error == "RATE_LIMITED":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="wait_or_degrade",
            retryable=False,
            fallback_tools=["use_cache", "answer_with_gap"],
            message="工具被限流，短时间内不应继续高频调用。",
        )
    if error == "CIRCUIT_OPEN":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="use_fallback_tool_or_answer_with_gap",
            retryable=False,
            fallback_tools=["use_cache", "fallback_tool", "answer_with_gap"],
            message="工具熔断中，请使用备用工具、缓存或带缺口回答。",
        )
    if error == "PROVIDER_ERROR":
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="retry_or_switch_provider",
            retryable=True,
            fallback_tools=["fallback_model", "template_answer"],
            message="模型服务异常，可有限重试或切换备用模型。",
        )
    if error in {"NOT_FOUND", "PERMISSION_DENIED"}:
        return ToolRecoveryAction(
            error_type=error,
            tool_kind=kind,
            action="ask_user_or_human_review",
            retryable=False,
            fallback_tools=["ask_human", "answer_with_gap"],
            message="资源不存在或权限不足，应澄清、人工介入或带缺口回答。",
        )
    return ToolRecoveryAction(
        error_type=error,
        tool_kind=kind,
        action="classify_error_and_answer_with_gap",
        retryable=False,
        fallback_tools=["answer_with_gap"],
        message="工具失败，请根据结构化错误决定是否换工具、重规划或带缺口回答。",
    )


def infer_tool_kind(tool_name: str) -> str:
    if tool_name in {"draft_llm_sql", "repair_sql", "emit_question_understanding", "draft_sql", "llm_chat", "llm_tool_chat"}:
        return "llm"
    if tool_name in {"execute_sql", "check_freshness", "doris_query"}:
        return "doris"
    if tool_name.startswith("semantic_") or tool_name.startswith("artifact_"):
        return "semantic"
    if tool_name.startswith("es_") or tool_name == "retrieve_knowledge":
        return "es"
    return "tool"


def structured_tool_message(
    tool_name: str,
    status: str,
    error_type: str = "",
    message: str = "",
    recovery: ToolRecoveryAction | None = None,
    circuit_key: str = "",
    retryable: bool | None = None,
    fallback_tools: Optional[List[str]] = None,
    tool_call_id: str = "",
    error_code: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    action = recovery or recovery_action_for(tool_name, error_type)
    code = error_code or error_type
    payload = {
        "toolName": tool_name,
        "status": status,
        "errorType": error_type,
        "errorCode": code,
        "retryable": action.retryable if retryable is None else bool(retryable),
        "recommendedAction": action.action,
        "fallbackTools": list(fallback_tools if fallback_tools is not None else action.fallback_tools),
        "message": message or action.message,
        "circuitKey": circuit_key,
    }
    if tool_call_id:
        payload["toolCallId"] = tool_call_id
    if details:
        payload["details"] = {key: value for key, value in details.items() if value not in ("", None, [], {})}
    return payload


def tool_error_details(
    tool_name: str,
    args: Dict[str, Any] | None = None,
    error_type: str = "",
    error_message: str = "",
    timeout_type: str = "",
    context: Dict[str, Any] | None = None,
    call_id: str = "",
) -> Dict[str, Any]:
    safe_args = args or {}
    kind = infer_tool_kind(tool_name)
    details: Dict[str, Any] = {
        "toolCallId": call_id,
        "toolName": tool_name,
        "toolKind": kind,
        "errorType": error_type,
        "paramsHash": (context or {}).get("paramsHash") or tool_params_hash(safe_args),
        "serviceName": (context or {}).get("serviceName") or kind,
        "target": (context or {}).get("target") or "",
        "circuitKey": (context or {}).get("circuitKey") or "",
        "timeoutType": timeout_type,
    }
    if kind == "doris":
        sql = str(safe_args.get("sql") or safe_args.get("query") or "")
        if sql:
            details["sqlHash"] = hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]
            details["sqlPreview"] = sql[:500]
        for key in ["queryId", "query_id", "taskId", "task_id", "dorisErrorCode", "sqlState", "failedStage"]:
            if safe_args.get(key):
                details[key] = safe_args.get(key)
    elif tool_name.startswith("artifact_"):
        for key in ["path", "relativePath", "merchantUri", "pattern", "offset", "limit"]:
            if safe_args.get(key):
                details[key] = safe_args.get(key)
    elif kind == "semantic":
        for key in ["refId", "ref_id", "assetType", "asset_type", "path", "relativePath", "merchantUri", "query"]:
            if safe_args.get(key):
                details[key] = safe_args.get(key)
    elif kind == "llm":
        for key in ["model", "provider", "promptId", "promptVersion", "templateFingerprint", "renderFingerprint"]:
            if safe_args.get(key):
                details[key] = safe_args.get(key)
    if error_message:
        details["messagePreview"] = str(error_message)[:500]
    return {key: value for key, value in details.items() if value not in ("", None, [], {})}


class CacheStore:
    def get(self, key: str) -> Any:
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError

    def clear(self) -> None:
        raise NotImplementedError

    def trace(self) -> Dict[str, Any]:
        return {}


class MemoryCacheStore(CacheStore):
    def __init__(self, max_entries: int = 512):
        self.max_entries = max(1, int(max_entries or 512))
        self._items: Dict[str, tuple[float, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        with self._lock:
            item = self._items.get(key)
            if not item:
                self._misses += 1
                return None
            expires_at, value = item
            if expires_at and expires_at < time.time():
                self._items.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        with self._lock:
            if len(self._items) >= self.max_entries:
                oldest = next(iter(self._items.keys()), "")
                if oldest:
                    self._items.pop(oldest, None)
            expires_at = time.time() + max(1, int(ttl_seconds or 1))
            self._items[key] = (expires_at, value)

    def delete(self, key: str) -> None:
        with self._lock:
            self._items.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def trace(self) -> Dict[str, Any]:
        with self._lock:
            return {"entries": len(self._items), "hits": self._hits, "misses": self._misses, "maxEntries": self.max_entries}


class RedisCacheStore(CacheStore):
    """Redis implementation for tool-runtime cache, with memory fallback."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.namespace = safe_redis_namespace(settings.redis_namespace)
        self._fallback = MemoryCacheStore(settings.cache_memory_max_entries)
        self._client = None
        self.available = False
        self.last_error = ""
        self._hits = 0
        self._misses = 0
        self._sets = 0
        try:
            import redis

            timeout = max(0.05, float(settings.redis_socket_timeout_seconds or 1.0))
            self._client = redis.Redis.from_url(
                settings.redis_url,
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

    def get(self, key: str) -> Any:
        if not key:
            self._misses += 1
            return None
        if not self.available or self._client is None:
            return self._fallback.get(key)
        try:
            raw = self._client.get(self._key(key))
            if raw is None:
                self._misses += 1
                return None
            self._hits += 1
            return json_cache_loads(raw)
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False
            return self._fallback.get(key)

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        if not key:
            return
        ttl = max(1, int(ttl_seconds or 1))
        if not self.available or self._client is None:
            self._fallback.set(key, value, ttl)
            return
        try:
            self._client.setex(self._key(key), ttl, json_cache_dumps(value))
            self._sets += 1
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False
            self._fallback.set(key, value, ttl)

    def delete(self, key: str) -> None:
        self._fallback.delete(key)
        if not self.available or self._client is None:
            return
        try:
            self._client.delete(self._key(key))
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False

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
            "backend": "redis" if self.available else "redis+memory_fallback",
            "available": self.available,
            "namespace": self.namespace,
            "entries": self._entry_count(),
            "hits": self._hits + int(self._fallback.trace().get("hits", 0)),
            "misses": self._misses + int(self._fallback.trace().get("misses", 0)),
            "sets": self._sets,
            "fallback": self._fallback.trace() if not self.available else {},
        }
        if self.last_error:
            trace["lastError"] = self.last_error
        return trace

    def _key(self, key: str) -> str:
        return "%s:tool_cache:%s" % (self.namespace, key)

    def _entry_count(self) -> int:
        if not self.available or self._client is None:
            return int(self._fallback.trace().get("entries", 0))
        try:
            count = 0
            for _ in self._client.scan_iter(match=self._key("*"), count=200):
                count += 1
                if count >= self.settings.cache_memory_max_entries:
                    break
            return count
        except Exception as exc:
            self.last_error = str(exc)[:200]
            return 0


class RateLimitStore:
    def allow(self, key: str, qps: int) -> bool:
        raise NotImplementedError

    def trace(self) -> Dict[str, Any]:
        return {}


class MemoryRateLimitStore(RateLimitStore):
    def __init__(self):
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._limited: Dict[str, int] = {}
        self._lock = threading.RLock()

    def allow(self, key: str, qps: int) -> bool:
        capacity = max(1, int(qps or 1))
        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key) or {"tokens": float(capacity), "updated": now}
            elapsed = max(0.0, now - float(bucket.get("updated") or now))
            tokens = min(float(capacity), float(bucket.get("tokens") or 0) + elapsed * capacity)
            allowed = tokens >= 1.0
            if allowed:
                tokens -= 1.0
            else:
                self._limited[key] = self._limited.get(key, 0) + 1
            bucket["tokens"] = tokens
            bucket["updated"] = now
            self._buckets[key] = bucket
            return allowed

    def trace(self) -> Dict[str, Any]:
        with self._lock:
            return {
                key: {"tokens": round(value.get("tokens", 0), 3), "limited": self._limited.get(key, 0)}
                for key, value in sorted(self._buckets.items())
            }


class RedisRateLimitStore(RateLimitStore):
    """Redis-backed token bucket for cross-instance tool rate limiting."""

    _ALLOW_SCRIPT = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local now_ms = tonumber(ARGV[2])
local ttl_ms = tonumber(ARGV[3])
local values = redis.call('HMGET', key, 'tokens', 'updated', 'limited', 'label')
local tokens = tonumber(values[1])
local updated = tonumber(values[2])
local limited = tonumber(values[3]) or 0
if tokens == nil then tokens = capacity end
if updated == nil then updated = now_ms end
local elapsed = math.max(0, now_ms - updated) / 1000
tokens = math.min(capacity, tokens + elapsed * capacity)
local allowed = 0
if tokens >= 1 then
  tokens = tokens - 1
  allowed = 1
else
  limited = limited + 1
end
redis.call('HSET', key, 'tokens', tokens, 'updated', now_ms, 'limited', limited, 'label', ARGV[4])
redis.call('PEXPIRE', key, ttl_ms)
return {allowed, tokens, limited}
"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.namespace = safe_redis_namespace(settings.redis_namespace)
        self._fallback = MemoryRateLimitStore()
        self._client = None
        self.available = False
        self.last_error = ""
        try:
            import redis

            timeout = max(0.05, float(settings.redis_socket_timeout_seconds or 1.0))
            self._client = redis.Redis.from_url(
                settings.redis_url,
                socket_timeout=timeout,
                socket_connect_timeout=timeout,
                decode_responses=True,
            )
            self._client.ping()
            self.available = True
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self._client = None
            self.available = False

    def allow(self, key: str, qps: int) -> bool:
        if not self.available or self._client is None:
            return self._fallback.allow(key, qps)
        capacity = max(1, int(qps or 1))
        redis_key = self._key(key)
        ttl_ms = max(1000, int((capacity + 2) * 1000))
        try:
            result = self._client.eval(self._ALLOW_SCRIPT, 1, redis_key, capacity, int(time.time() * 1000), ttl_ms, key)
            return bool(int(result[0]))
        except Exception as exc:
            self.last_error = str(exc)[:200]
            self.available = False
            return self._fallback.allow(key, qps)

    def trace(self) -> Dict[str, Any]:
        if not self.available or self._client is None:
            payload = self._fallback.trace()
            payload["_backend"] = "redis+memory_fallback"
            payload["_available"] = False
            if self.last_error:
                payload["_lastError"] = self.last_error
            return payload
        trace: Dict[str, Any] = {"_backend": "redis", "_available": True, "_namespace": self.namespace}
        try:
            for redis_key in self._client.scan_iter(match=self._key("*"), count=100):
                data = self._client.hgetall(redis_key) or {}
                label = data.get("label") or str(redis_key).rsplit(":", 1)[-1]
                trace[str(label)] = {
                    "tokens": round(float(data.get("tokens") or 0), 3),
                    "limited": int(float(data.get("limited") or 0)),
                }
                if len(trace) > 50:
                    break
        except Exception as exc:
            self.last_error = str(exc)[:200]
            trace["_lastError"] = self.last_error
        return trace

    def _key(self, key: str) -> str:
        digest = hashlib.sha256(str(key or "").encode("utf-8")).hexdigest()[:24]
        return "%s:rate_limit:%s" % (self.namespace, digest)


class RoundRobinLoadBalancer:
    def __init__(self, targets_by_kind: Optional[Dict[str, List[LoadBalancerTarget]]] = None):
        self.targets_by_kind = targets_by_kind or {}
        self._cursor: Dict[str, int] = {}
        self._lock = threading.RLock()

    @staticmethod
    def parse_targets(raw: str, default_name: str = "") -> List[LoadBalancerTarget]:
        targets: List[LoadBalancerTarget] = []
        for index, entry in enumerate([item.strip() for item in (raw or "").split(",") if item.strip()]):
            name = default_name or "target_%d" % (index + 1)
            endpoint = entry
            if "=" in entry:
                name, endpoint = entry.split("=", 1)
            targets.append(LoadBalancerTarget(name=name.strip() or "target_%d" % (index + 1), endpoint=endpoint.strip(), healthy=True))
        return targets

    def select(self, kind: str) -> LoadBalancerTarget:
        candidates = [target for target in self.targets_by_kind.get(kind, []) if target.healthy]
        if not candidates:
            return LoadBalancerTarget(name="default", endpoint="", healthy=True)
        with self._lock:
            cursor = self._cursor.get(kind, 0)
            target = candidates[cursor % len(candidates)]
            self._cursor[kind] = cursor + 1
            return target

    def trace(self) -> Dict[str, Any]:
        with self._lock:
            return {
                kind: {
                    "targets": [target.model_dump(by_alias=True) for target in targets],
                    "cursor": self._cursor.get(kind, 0),
                }
                for kind, targets in self.targets_by_kind.items()
            }


class RuntimeMetricsAggregator:
    def __init__(self):
        self._metrics: Dict[str, ToolRuntimeMetrics] = {}
        self._durations: Dict[str, List[int]] = {}
        self._lock = threading.RLock()

    def record(
        self,
        tool_name: str,
        status: str,
        duration_ms: int,
        error_type: str = "",
        cache_hit: bool = False,
        rate_limited: bool = False,
        circuit_blocked: bool = False,
        attempts: int = 0,
        target: str = "",
    ) -> None:
        with self._lock:
            metric = self._metrics.get(tool_name) or ToolRuntimeMetrics(tool_name=tool_name)
            metric.calls += 1
            if status == "success":
                metric.successes += 1
            elif rate_limited:
                metric.rate_limited += 1
            elif circuit_blocked:
                metric.circuit_blocked += 1
            else:
                metric.failures += 1
            if error_type == "TIMEOUT":
                metric.timeouts += 1
            if cache_hit:
                metric.cache_hits += 1
            else:
                metric.cache_misses += 1
            metric.retries += max(0, int(attempts or 0) - 1)
            metric.total_duration_ms += max(0, int(duration_ms or 0))
            metric.last_error_type = error_type or metric.last_error_type
            metric.last_target = target or metric.last_target
            durations = self._durations.setdefault(tool_name, [])
            durations.append(max(0, int(duration_ms or 0)))
            if len(durations) > 200:
                del durations[: len(durations) - 200]
            metric.p95_duration_ms = percentile(durations, 0.95)
            metric.p99_duration_ms = percentile(durations, 0.99)
            self._metrics[tool_name] = metric

    def snapshot(self) -> List[ToolRuntimeMetrics]:
        with self._lock:
            return [item.model_copy(deep=True) for item in self._metrics.values()]

    def trace(self) -> Dict[str, Any]:
        return {"tools": [item.model_dump(by_alias=True) for item in self.snapshot()]}


def percentile(values: List[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[max(0, min(index, len(ordered) - 1))]


class RuntimeAlertManager:
    def __init__(self, settings: Settings, metrics: RuntimeMetricsAggregator):
        self.settings = settings
        self.metrics = metrics
        self._alerts: List[RuntimeAlert] = []
        self._seen: set[str] = set()
        self._lock = threading.RLock()

    def evaluate(self) -> List[RuntimeAlert]:
        threshold = max(1, int(getattr(self.settings, "alert_p95_threshold_ms", 30000) or 30000))
        created: List[RuntimeAlert] = []
        for metric in self.metrics.snapshot():
            if metric.p95_duration_ms >= threshold:
                created.append(self._create("TOOL_P95_HIGH", metric.tool_name, metric.p95_duration_ms, threshold))
            if metric.timeouts >= 3 and metric.calls:
                created.append(self._create("TOOL_TIMEOUTS_HIGH", metric.tool_name, metric.timeouts, 3))
            if metric.circuit_blocked > 0:
                created.append(self._create("TOOL_CIRCUIT_OPEN", metric.tool_name, metric.circuit_blocked, 1))
        return [item for item in created if item.alert_id]

    def _create(self, code: str, tool_name: str, value: float, threshold: float) -> RuntimeAlert:
        key = "%s:%s:%s" % (code, tool_name, int(value))
        with self._lock:
            if key in self._seen:
                return RuntimeAlert()
            self._seen.add(key)
            alert = RuntimeAlert(
                alert_id=hashlib.sha256(key.encode("utf-8")).hexdigest()[:16],
                severity="warning",
                code=code,
                tool_name=tool_name,
                value=value,
                threshold=threshold,
                message="%s %s reached %s (threshold %s)" % (tool_name, code, value, threshold),
                created_at=datetime.now().isoformat(),
            )
            self._alerts.append(alert)
            self._send_webhook(alert)
            return alert

    def _send_webhook(self, alert: RuntimeAlert) -> None:
        if not self.settings.alert_webhook_url:
            return
        try:
            data = json.dumps(alert.model_dump(by_alias=True), ensure_ascii=False, default=str).encode("utf-8")
            req = url_request.Request(
                self.settings.alert_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            url_request.urlopen(req, timeout=2).close()
        except Exception:
            return

    def trace(self) -> List[Dict[str, Any]]:
        self.evaluate()
        with self._lock:
            return [item.model_dump(by_alias=True) for item in self._alerts[-50:]]


class ToolRuntimePolicyRegistry:
    """Timeout/retry profiles for tools, mirroring DeerFlow's runtime boundaries."""

    def __init__(self, settings: Settings):
        self.settings = settings

    def policy_for(self, tool_name: str) -> ToolRuntimePolicy:
        if tool_name in {"inspect_schema", "resolve_columns", "check_freshness", "summarize_node_result"}:
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=5,
                max_retries=0,
                non_retryable_errors=["UNKNOWN_COLUMN", "INVALID_ARGUMENT"],
                fallback_tools=["semantic_read", "answer_with_gap"],
            )
        if tool_name in {"semantic_ls", "semantic_read", "semantic_grep", "semantic_write", "artifact_ls", "artifact_read", "artifact_grep", "artifact_write"}:
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=5,
                max_retries=0,
                non_retryable_errors=["INVALID_ARGUMENT", "ARTIFACT_NOT_FOUND", "SEMANTIC_REF_NOT_FOUND", "TOOL_CONTRACT_VIOLATION"],
                fallback_tools=["semantic_grep", "retrieve_knowledge", "answer_with_gap"],
            )
        if tool_name in {"draft_llm_sql", "repair_sql", "emit_question_understanding", "draft_sql"}:
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=max(1, self.settings.llm_request_timeout_seconds),
                max_retries=1,
                backoff_seconds=0.2,
                retryable_errors=["TIMEOUT", "PROVIDER_ERROR"],
                fallback_tools=["structured_sql_fallback", "fallback_model"],
            )
        if tool_name == "execute_sql":
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=max(1, self.settings.doris_read_timeout_seconds),
                max_retries=0,
                backoff_seconds=0.0,
                retryable_errors=[],
                non_retryable_errors=["UNKNOWN_COLUMN", "MEM_ALLOC_FAILED", "TIMEOUT", "PARSE_ERROR", "UNSAFE_SQL", "UNKNOWN_BASE_TABLE", "TOOL_CONTRACT_VIOLATION"],
                fallback_tools=["use_cache", "structured_sql_fallback", "partial_answer", "answer_with_gap"],
            )
        if tool_name in {"node_agent", "node_agent_batch"}:
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
                max_retries=0,
                non_retryable_errors=["TIMEOUT"],
                fallback_tools=["skip_non_critical_node", "partial_answer"],
            )
        return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=max(1, self.settings.agent_node_timeout_seconds), max_retries=0)

    def trace(self) -> List[Dict[str, Any]]:
        names = [
            "inspect_schema",
            "draft_llm_sql",
            "repair_sql",
            "execute_sql",
            "node_agent",
            "node_agent_batch",
            "emit_question_understanding",
            "semantic_read",
            "artifact_read",
        ]
        return [self.policy_for(name).model_dump(by_alias=True) for name in names]


class ToolFailureRegistry:
    """Track repeated tool failures and open lightweight circuit breakers."""

    def __init__(self, repeat_threshold: int = 2, circuit_threshold: int = 5, cooldown_seconds: int = 60):
        self.repeat_threshold = repeat_threshold
        self.circuit_threshold = circuit_threshold
        self.cooldown_seconds = max(1, int(cooldown_seconds or 60))
        self.records: Dict[str, ToolFailureRecord] = {}
        self.circuits: Dict[str, CircuitBreakerState] = {}
        self._lock = threading.RLock()

    def fingerprint(
        self,
        tool_name: str,
        args: Any,
        service_name: str = "",
        target: str = "",
        merchant_id: str = "",
        thread_id: str = "",
    ) -> str:
        context = self.context(tool_name, args, service_name, target, merchant_id, thread_id)
        return context["fingerprint"]

    def context(
        self,
        tool_name: str,
        args: Any,
        service_name: str = "",
        target: str = "",
        merchant_id: str = "",
        thread_id: str = "",
    ) -> Dict[str, str]:
        scope = current_tool_runtime_scope()
        context = normalize_tool_context(tool_name, args, service_name, target)
        context["merchantId"] = str(merchant_id or context.get("merchantId") or scope.get("merchantId") or "")
        context["threadId"] = str(thread_id or context.get("threadId") or scope.get("threadId") or "")
        run_id = str(scope.get("runId") or "")
        context["circuitKey"] = "tool=%s|service=%s|target=%s|merchant=%s|thread=%s|run=%s" % (
            tool_name,
            context["serviceName"] or "tool",
            context["target"] or "default",
            context["merchantId"] or "*",
            context["threadId"] or "*",
            run_id or "*",
        )
        context["fingerprint"] = "%s|params=%s" % (context["circuitKey"], context["paramsHash"])
        return context

    def should_block(
        self,
        tool_name: str,
        args: Any,
        service_name: str = "",
        target: str = "",
        merchant_id: str = "",
        thread_id: str = "",
    ) -> CircuitBreakerState | None:
        context = self.context(tool_name, args, service_name, target, merchant_id, thread_id)
        with self._lock:
            circuit = self.circuits.get(context["circuitKey"])
            if circuit and circuit.state == "open":
                if circuit.open_until_ms and circuit.open_until_ms <= now_ms():
                    circuit.state = "half_open"
                    circuit.open = False
                    circuit.half_open_probe_in_flight = True
                    circuit.last_probe_at_ms = now_ms()
                    circuit.reason = "half-open probe allowed after cooldown"
                    self.circuits[context["circuitKey"]] = circuit
                else:
                    return circuit
            elif circuit and circuit.state == "half_open" and circuit.half_open_probe_in_flight:
                circuit.open = True
                circuit.reason = circuit.reason or "half-open probe already in flight"
                return circuit
            record = self.records.get(context["fingerprint"])
            if record and record.blocked:
                return CircuitBreakerState(
                    circuit_key=context["fingerprint"],
                    tool_name=tool_name,
                    service_name=context["serviceName"],
                    target=context["target"],
                    merchant_id=context["merchantId"],
                    thread_id=context["threadId"],
                    params_hash=context["paramsHash"],
                    state="open",
                    open=True,
                    failure_count=record.count,
                    reason="repeated identical failure: %s" % record.error_type,
                    recovery_action=record.recovery_action,
                    recommended_action=record.recovery_action,
                )
            return None

    def record_success(
        self,
        tool_name: str,
        args: Any,
        service_name: str = "",
        target: str = "",
        merchant_id: str = "",
        thread_id: str = "",
    ) -> CircuitBreakerState | None:
        context = self.context(tool_name, args, service_name, target, merchant_id, thread_id)
        with self._lock:
            self.records.pop(context["fingerprint"], None)
            circuit = self.circuits.get(context["circuitKey"])
            if circuit:
                circuit.failure_count = 0
                circuit.open = False
                circuit.state = "closed"
                circuit.reason = ""
                circuit.open_until_ms = 0
                circuit.half_open_probe_in_flight = False
                self.circuits[context["circuitKey"]] = circuit
                return circuit
            return None

    def record_failure(
        self,
        tool_name: str,
        args: Any,
        error_type: str,
        error_message: str,
        service_name: str = "",
        target: str = "",
        merchant_id: str = "",
        thread_id: str = "",
    ) -> ToolFailureRecord:
        context = self.context(tool_name, args, service_name, target, merchant_id, thread_id)
        fingerprint = context["fingerprint"]
        recovery = recovery_action_for(tool_name, error_type, context["serviceName"])
        timestamp = now_ms()
        with self._lock:
            record = self.records.get(fingerprint) or ToolFailureRecord(
                fingerprint=fingerprint,
                tool_name=tool_name,
                service_name=context["serviceName"],
                target=context["target"],
                merchant_id=context["merchantId"],
                thread_id=context["threadId"],
                params_hash=context["paramsHash"],
                circuit_key=context["circuitKey"],
                first_failed_at_ms=timestamp,
            )
            record.error_type = error_type or "ERROR"
            record.error_message = str(error_message or "")[:500]
            record.count += 1
            record.blocked = record.count >= self.repeat_threshold
            record.last_failed_at_ms = timestamp
            record.recovery_action = recovery.action
            self.records[fingerprint] = record
            circuit = self.circuits.get(context["circuitKey"]) or CircuitBreakerState(
                circuit_key=context["circuitKey"],
                tool_name=tool_name,
                service_name=context["serviceName"],
                target=context["target"],
                merchant_id=context["merchantId"],
                thread_id=context["threadId"],
                params_hash=context["paramsHash"],
            )
            circuit.failure_count += 1
            circuit.params_hash = context["paramsHash"]
            circuit.recovery_action = recovery.action
            circuit.recommended_action = recovery.action
            circuit.fallback_tools = list(recovery.fallback_tools)
            if circuit.state == "half_open" or circuit.failure_count >= self.circuit_threshold:
                circuit.state = "open"
                circuit.open = True
                circuit.reason = "tool failure threshold reached: %s" % record.error_type
                circuit.opened_at_ms = now_ms()
                circuit.open_until_ms = circuit.opened_at_ms + self.cooldown_seconds * 1000
                circuit.half_open_probe_in_flight = False
            else:
                circuit.state = "closed"
                circuit.open = False
            self.circuits[context["circuitKey"]] = circuit
            return record

    def trace(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "failures": [item.model_dump(by_alias=True) for item in self.records.values()],
                "circuits": [item.model_dump(by_alias=True) for item in self.circuits.values()],
            }


def default_load_balancer(settings: Settings) -> RoundRobinLoadBalancer:
    targets = {
        "llm": RoundRobinLoadBalancer.parse_targets(settings.llm_targets, "llm"),
        "es": RoundRobinLoadBalancer.parse_targets(settings.es_targets, "es"),
        "doris": RoundRobinLoadBalancer.parse_targets(settings.doris_targets, "doris"),
    }
    return RoundRobinLoadBalancer({kind: values for kind, values in targets.items() if values})


def default_cache_store(settings: Settings) -> CacheStore:
    if settings.redis_enabled and settings.redis_cache_enabled:
        return RedisCacheStore(settings)
    return MemoryCacheStore(settings.cache_memory_max_entries)


def default_rate_limit_store(settings: Settings) -> RateLimitStore:
    if settings.redis_enabled and settings.redis_rate_limit_enabled:
        return RedisRateLimitStore(settings)
    return MemoryRateLimitStore()


def safe_redis_namespace(namespace: str) -> str:
    text = str(namespace or "merchant_ai").strip()
    return "".join(ch if ch.isalnum() or ch in {"_", "-", ":"} else "_" for ch in text) or "merchant_ai"


class ToolRuntimeService:
    """Enterprise-style gateway for native tools without changing tool handlers."""

    def __init__(
        self,
        settings: Settings,
        policy_registry: ToolRuntimePolicyRegistry | None = None,
        failure_registry: ToolFailureRegistry | None = None,
        cache_store: CacheStore | None = None,
        rate_limit_store: RateLimitStore | None = None,
        load_balancer: RoundRobinLoadBalancer | None = None,
        metrics: RuntimeMetricsAggregator | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        self.settings = settings
        self.policy_registry = policy_registry or ToolRuntimePolicyRegistry(settings)
        self.failure_registry = failure_registry or ToolFailureRegistry(
            repeat_threshold=settings.tool_failure_repeat_threshold,
            circuit_threshold=settings.tool_circuit_threshold,
            cooldown_seconds=settings.tool_circuit_cooldown_seconds,
        )
        self.cache_store = cache_store or default_cache_store(settings)
        self.rate_limit_store = rate_limit_store or default_rate_limit_store(settings)
        self.load_balancer = load_balancer or default_load_balancer(settings)
        self.metrics = metrics or RuntimeMetricsAggregator()
        self.alert_manager = RuntimeAlertManager(settings, self.metrics)
        self.tool_registry = tool_registry or canonical_tool_registry()
        self._events: List[Dict[str, Any]] = []
        self._events_lock = threading.RLock()

    def cache_key(self, tool_name: str, args: Dict[str, Any], policy: ToolCachePolicy | None = None) -> str:
        cache_policy = policy or ToolCachePolicy()
        source: Dict[str, Any]
        if cache_policy.key_fields:
            source = {field: args.get(field) for field in cache_policy.key_fields}
        else:
            source = dict(args or {})
        source.setdefault("semanticVersion", args.get("semanticVersion") or args.get("semantic_version") or "")
        try:
            raw = json.dumps(source, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            raw = str(source)
        namespace = cache_policy.namespace or tool_name
        return "%s:%s" % (namespace, hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24])

    def execute(
        self,
        tool_name: str,
        args: Dict[str, Any],
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        call_id: str = "",
        cache_policy: ToolCachePolicy | None = None,
        target_kind: str = "",
        cancel_event: Any = None,
    ) -> ToolCallExecutionResult:
        call = ToolCallRequest(id=call_id or tool_name, name=tool_name, args=args or {})
        service_name = target_kind or self._tool_kind(tool_name)
        selected_target = self.load_balancer.select(service_name)
        context = self.failure_registry.context(tool_name, call.args, service_name=service_name, target=selected_target.name)
        idempotency_key = tool_idempotency_key(tool_name, call.id, context)
        blocked = self.failure_registry.should_block(tool_name, call.args, service_name=service_name, target=selected_target.name)
        pre_runtime_events: List[Dict[str, Any]] = [
            self._record_event(
                "tool.started",
                ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="started",
                    idempotency_key=idempotency_key,
                    params_hash=context["paramsHash"],
                    target=selected_target.name,
                    service_name=service_name,
                    circuit_key=context["circuitKey"],
                ),
                {"idempotencyKey": idempotency_key, "paramsHash": context["paramsHash"]},
            )
        ]
        if blocked:
            recovery = recovery_action_for(tool_name, "CIRCUIT_OPEN", service_name)
            details = tool_error_details(tool_name, call.args, "CIRCUIT_OPEN", blocked.reason, context=context, call_id=call.id)
            tool_message = structured_tool_message(
                tool_name,
                "blocked",
                "CIRCUIT_OPEN",
                blocked.reason,
                recovery,
                blocked.circuit_key or context["circuitKey"],
                tool_call_id=call.id,
                details=details,
            )
            result = ToolCallExecutionResult(
                id=call.id,
                name=tool_name,
                status="blocked",
                idempotency_key=idempotency_key,
                params_hash=context["paramsHash"],
                error_type="CIRCUIT_OPEN",
                error_message=blocked.reason,
                target=selected_target.name,
                service_name=service_name,
                circuit_key=blocked.circuit_key or context["circuitKey"],
                retryable=False,
                recommended_action=recovery.action,
                fallback_tools=list(recovery.fallback_tools),
                details=details,
                tool_message=tool_message,
            )
            result.runtime_events.extend(pre_runtime_events)
            result.runtime_events.append(self._record_event("tool.circuit.open", result, {"circuit": blocked.model_dump(by_alias=True)}))
            self.metrics.record(tool_name, result.status, 0, result.error_type, circuit_blocked=True)
            self.alert_manager.evaluate()
            return result
        probe_circuit = self.failure_registry.circuits.get(context["circuitKey"])
        if probe_circuit and probe_circuit.state == "half_open" and probe_circuit.half_open_probe_in_flight:
            pre_runtime_events.append(
                self._record_event(
                    "tool.circuit.half_open",
                    ToolCallExecutionResult(
                        id=call.id,
                        name=tool_name,
                        status="probing",
                        idempotency_key=idempotency_key,
                        params_hash=context["paramsHash"],
                        target=selected_target.name,
                        service_name=service_name,
                        circuit_key=context["circuitKey"],
                    ),
                    {"circuit": probe_circuit.model_dump(by_alias=True)},
                )
            )
        if self.settings.tool_rate_limit_enabled:
            rate_key = scoped_rate_limit_key(
                tool_name,
                service_name,
                selected_target.name,
                context.get("merchantId", ""),
                context.get("threadId", ""),
            )
            if not self.rate_limit_store.allow(rate_key, self._tool_qps(tool_name, service_name)):
                recovery = recovery_action_for(tool_name, "RATE_LIMITED", service_name)
                details = tool_error_details(tool_name, call.args, "RATE_LIMITED", "tool rate limited: %s" % rate_key, context=context, call_id=call.id)
                tool_message = structured_tool_message(
                    tool_name,
                    "blocked",
                    "RATE_LIMITED",
                    "tool rate limited: %s" % rate_key,
                    recovery,
                    context["circuitKey"],
                    tool_call_id=call.id,
                    details=details,
                )
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="blocked",
                    idempotency_key=idempotency_key,
                    params_hash=context["paramsHash"],
                    error_type="RATE_LIMITED",
                    error_message="tool rate limited: %s" % rate_key,
                    rate_limited=True,
                    target=selected_target.name,
                    service_name=service_name,
                    circuit_key=context["circuitKey"],
                    retryable=False,
                    recommended_action=recovery.action,
                    fallback_tools=list(recovery.fallback_tools),
                    details=details,
                    tool_message=tool_message,
                    runtime_events=list(pre_runtime_events),
                )
                event = self._record_event("tool.rate_limited", result, {"rateLimitKey": rate_key})
                result.runtime_events.append(event)
                self.metrics.record(tool_name, result.status, 0, result.error_type, rate_limited=True)
                return result
        cache_key = ""
        if cache_policy and cache_policy.enabled and self.settings.cache_enabled:
            cache_key = self.cache_key(tool_name, call.args, cache_policy)
            cached = self.cache_store.get(cache_key)
            if cached is not None:
                closed_circuit = None
                if probe_circuit and probe_circuit.state == "half_open":
                    closed_circuit = self.failure_registry.record_success(tool_name, call.args, service_name=service_name, target=selected_target.name)
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="success",
                    result=cached if isinstance(cached, dict) else {"value": cached},
                    idempotency_key=idempotency_key,
                    params_hash=context["paramsHash"],
                    cache_hit=True,
                    cache_key=cache_key,
                    attempts=0,
                    target=selected_target.name,
                    service_name=service_name,
                    circuit_key=context["circuitKey"],
                    tool_message={"toolName": tool_name, "status": "success", "cacheHit": True},
                    runtime_events=list(pre_runtime_events),
                )
                if closed_circuit:
                    result.runtime_events.append(self._record_event("tool.circuit.closed", result, {"circuit": closed_circuit.model_dump(by_alias=True)}))
                contract = validate_tool_result_contract(tool_name, result.result, self.tool_registry)
                result.contract = contract
                result.result_hash = str(contract.get("resultHash") or "")
                self.metrics.record(tool_name, result.status, 0, cache_hit=True)
                return result
        started = time.monotonic()
        policy = self.policy_registry.policy_for(tool_name)
        attempts = max(1, policy.max_retries + 1)
        last_error = ""
        last_error_type = "ERROR"
        last_timeout_type = ""
        all_attempt_events: List[Dict[str, Any]] = []
        for attempt in range(attempts):
            attempt_events: List[Dict[str, Any]] = []
            try:
                next_args = dict(call.args)
                if selected_target.endpoint:
                    next_args.setdefault("_target", selected_target.model_dump(by_alias=True))
                value = self._call_with_timeout(
                    handler,
                    next_args,
                    policy.timeout_seconds,
                    cancel_event=cancel_event,
                    heartbeat=lambda elapsed_ms, timeout_seconds: attempt_events.append(
                        self._record_event(
                            "tool.heartbeat",
                            ToolCallExecutionResult(
                                id=call.id,
                                name=tool_name,
                                status="running",
                                idempotency_key=idempotency_key,
                                params_hash=context["paramsHash"],
                                target=selected_target.name,
                                service_name=service_name,
                                circuit_key=context["circuitKey"],
                            ),
                            {"elapsedMs": elapsed_ms, "timeoutSeconds": timeout_seconds, "attempt": attempt + 1},
                        )
                    ),
                )
                duration_ms = int((time.monotonic() - started) * 1000)
                closed_circuit = self.failure_registry.record_success(tool_name, call.args, service_name=service_name, target=selected_target.name)
                contract = validate_tool_result_contract(tool_name, value or {}, self.tool_registry)
                if not contract.get("valid", True) and tool_contract_enforced(contract):
                    raise RuntimeError("TOOL_CONTRACT_VIOLATION missing=%s" % ",".join(contract.get("missingKeys") or []))
                if cache_key and cache_policy:
                    self.cache_store.set(cache_key, value or {}, cache_policy.ttl_seconds or self.settings.semantic_cache_ttl_seconds)
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="success",
                    result=value or {},
                    idempotency_key=idempotency_key,
                    params_hash=context["paramsHash"],
                    duration_ms=duration_ms,
                    attempts=attempt + 1,
                    cache_hit=False,
                    cache_key=cache_key,
                    target=selected_target.name,
                    service_name=service_name,
                    circuit_key=context["circuitKey"],
                    contract=contract,
                    result_hash=str(contract.get("resultHash") or ""),
                    tool_message={"toolName": tool_name, "status": "success", "cacheHit": False},
                    runtime_events=list(pre_runtime_events) + all_attempt_events + attempt_events,
                )
                if closed_circuit and closed_circuit.state == "closed":
                    event = self._record_event("tool.circuit.closed", result, {"circuit": closed_circuit.model_dump(by_alias=True)})
                    result.runtime_events.append(event)
                self.metrics.record(tool_name, result.status, duration_ms, attempts=result.attempts, target=selected_target.name)
                self.alert_manager.evaluate()
                return result
            except Exception as exc:
                all_attempt_events.extend(attempt_events)
                last_error = str(exc)
                last_error_type = classify_tool_error(exc)
                last_timeout_type = classify_timeout_type(exc, service_name) if last_error_type == "TIMEOUT" else ""
                if last_error_type in policy.non_retryable_errors or attempt >= attempts - 1:
                    break
                if policy.retryable_errors and last_error_type not in policy.retryable_errors:
                    break
                if policy.backoff_seconds > 0:
                    time.sleep(policy.backoff_seconds * (attempt + 1))
                self._record_event(
                    "tool.retry",
                    ToolCallExecutionResult(
                        id=call.id,
                        name=tool_name,
                        status="retrying",
                        idempotency_key=idempotency_key,
                        params_hash=context["paramsHash"],
                        error_type=last_error_type,
                        error_message=last_error[:500],
                        timeout_type=last_timeout_type,
                        attempts=attempt + 1,
                        target=selected_target.name,
                        service_name=service_name,
                        circuit_key=context["circuitKey"],
                    ),
                    {"nextAttempt": attempt + 2},
                )
        duration_ms = int((time.monotonic() - started) * 1000)
        record = self.failure_registry.record_failure(tool_name, call.args, last_error_type, last_error, service_name=service_name, target=selected_target.name)
        recovery = recovery_action_for(tool_name, last_error_type, service_name)
        details = tool_error_details(tool_name, call.args, last_error_type, last_error[:500], timeout_type=last_timeout_type, context=context, call_id=call.id)
        tool_message = structured_tool_message(tool_name, "failed", last_error_type, last_error[:500], recovery, record.circuit_key, tool_call_id=call.id, details=details)
        result = ToolCallExecutionResult(
            id=call.id,
            name=tool_name,
            status="failed",
            idempotency_key=idempotency_key,
            params_hash=context["paramsHash"],
            error_type=last_error_type,
            error_message=last_error[:500],
            timeout_type=last_timeout_type,
            duration_ms=duration_ms,
            attempts=attempts,
            cache_key=cache_key,
            target=selected_target.name,
            service_name=service_name,
            circuit_key=record.circuit_key,
            retryable=recovery.retryable,
            recommended_action=recovery.action,
            fallback_tools=list(recovery.fallback_tools or policy.fallback_tools),
            details=details,
            tool_message=tool_message,
            runtime_events=list(pre_runtime_events) + all_attempt_events,
        )
        failed_event = self._record_event("tool.failed", result, {"failure": record.model_dump(by_alias=True)})
        result.runtime_events.append(failed_event)
        if record.recovery_action:
            result.runtime_events.append(self._record_event("tool.recovery.recommended", result, {"recommendedAction": record.recovery_action}))
        circuit = self.failure_registry.circuits.get(record.circuit_key)
        if circuit and circuit.open:
            result.runtime_events.append(self._record_event("tool.circuit.open", result, {"circuit": circuit.model_dump(by_alias=True)}))
        self.metrics.record(tool_name, result.status, duration_ms, result.error_type, attempts=result.attempts, target=selected_target.name)
        self.alert_manager.evaluate()
        return result

    def execute_many(
        self,
        tool_calls: Iterable[ToolCallRequest],
        handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        cache_policies: Dict[str, ToolCachePolicy] | None = None,
    ) -> List[ToolCallExecutionResult]:
        calls = list(tool_calls)
        if not calls:
            return []
        order = {call.id: index for index, call in enumerate(calls)}
        results: List[ToolCallExecutionResult] = []
        max_workers = min(max(1, self.settings.tool_max_concurrency), len(calls))
        batch_probe = ToolCallExecutionResult(id="tool_batch", name="execute_many", status="running", service_name="tool")
        self._record_event(
            "tool.parallel.batch_started",
            batch_probe,
            {"toolCallCount": len(calls), "maxConcurrency": max_workers, "toolNames": [call.name for call in calls]},
        )
        started = time.monotonic()
        executor = ThreadPoolExecutor(max_workers=max_workers)
        futures = {}
        try:
            for call in calls:
                handler = handlers.get(call.name)
                if handler is None:
                    results.append(ToolCallExecutionResult(id=call.id, name=call.name, status="failed", error_type="UNKNOWN_TOOL", error_message="No handler registered"))
                    continue
                futures[submit_with_current_context(
                    executor,
                    self.execute,
                    call.name,
                    call.args,
                    handler,
                    call.id,
                    (cache_policies or {}).get(call.name),
                )] = call
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    call = futures[future]
                    results.append(ToolCallExecutionResult(id=call.id, name=call.name, status="failed", error_type=classify_tool_error(exc), error_message=str(exc)[:500]))
        finally:
            for future in futures:
                if not future.done():
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        self._record_event(
            "tool.parallel.batch_finished",
            ToolCallExecutionResult(id="tool_batch", name="execute_many", status="finished", service_name="tool"),
            {
                "toolCallCount": len(calls),
                "maxConcurrency": max_workers,
                "durationMs": int((time.monotonic() - started) * 1000),
                "successCount": sum(1 for item in results if item.status == "success"),
                "failureCount": sum(1 for item in results if item.status != "success"),
            },
        )
        return sorted(results, key=lambda item: order.get(item.id, 999))

    def trace(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.trace(),
            "rateLimits": self.rate_limit_store.trace(),
            "cache": self.cache_store.trace(),
            "loadBalancer": self.load_balancer.trace(),
            "alerts": self.alert_manager.trace(),
            "events": self.events(),
        }

    def events(self) -> List[Dict[str, Any]]:
        with self._events_lock:
            return list(self._events[-200:])

    def _record_event(self, event_type: str, result: ToolCallExecutionResult, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        event = {
            "eventId": hashlib.sha256(("%s:%s:%s:%s" % (event_type, result.id, result.name, time.time_ns())).encode("utf-8")).hexdigest()[:16],
            "eventType": event_type,
            "toolCallId": result.id,
            "toolName": result.name,
            "status": result.status,
            "errorType": result.error_type,
            "timeoutType": result.timeout_type,
            "serviceName": result.service_name,
            "target": result.target,
            "circuitKey": result.circuit_key,
            "recommendedAction": result.recommended_action,
            "fallbackTools": result.fallback_tools,
            "createdAt": datetime.now().isoformat(),
            "payload": payload or {},
        }
        with self._events_lock:
            self._events.append(event)
            self._events = self._events[-300:]
        return event

    def _call_with_timeout(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        args: Dict[str, Any],
        timeout_seconds: int,
        cancel_event: Any = None,
        heartbeat: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, Any]:
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        execution_cancel_event = cancel_event or threading.Event()
        context = copy_context()

        def invoke_handler() -> None:
            try:
                with tool_cancel_scope(execution_cancel_event):
                    result_queue.put(("ok", handler(args)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=context.run, args=(invoke_handler,), name="tool-runtime-call", daemon=True)
        thread.start()
        timeout = max(1, int(timeout_seconds or 1))
        started = time.monotonic()
        heartbeat_interval = max(0.1, float(getattr(self.settings, "tool_heartbeat_interval_seconds", 5.0) or 5.0))
        next_heartbeat_at = started + heartbeat_interval
        try:
            while True:
                if bool(getattr(execution_cancel_event, "is_set", lambda: False)()):
                    raise RuntimeError("tool execution canceled")
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("tool execution timed out")
                wait_for = min(remaining, max(0.05, next_heartbeat_at - time.monotonic()))
                try:
                    status, value = result_queue.get(timeout=wait_for)
                    if status == "error":
                        raise value
                    return value
                except queue.Empty:
                    now = time.monotonic()
                    if heartbeat and now >= next_heartbeat_at:
                        heartbeat(int((now - started) * 1000), timeout)
                        next_heartbeat_at = now + heartbeat_interval
        except TimeoutError as exc:
            if hasattr(execution_cancel_event, "set"):
                execution_cancel_event.set()
            raise TimeoutError("tool execution timed out") from exc

    def _tool_kind(self, tool_name: str) -> str:
        if tool_name in {"draft_llm_sql", "repair_sql", "emit_question_understanding", "draft_sql", "llm_chat", "llm_tool_chat"}:
            return "llm"
        if tool_name in {"execute_sql", "check_freshness", "doris_query"}:
            return "doris"
        if tool_name.startswith("semantic_") or tool_name.startswith("artifact_"):
            return "semantic"
        if tool_name.startswith("es_") or tool_name == "retrieve_knowledge":
            return "es"
        return "tool"

    def _tool_qps(self, tool_name: str, kind: str) -> int:
        if kind == "llm":
            return max(1, self.settings.tool_default_qps)
        if kind == "doris":
            return max(1, self.settings.tool_default_qps)
        return max(1, self.settings.tool_default_qps)


class ToolCallExecutor:
    """Execute multiple native tool_calls with result pairing and failure isolation."""

    def __init__(self, policy_registry: ToolRuntimePolicyRegistry, failure_registry: ToolFailureRegistry, max_concurrency: int = 3):
        self.policy_registry = policy_registry
        self.failure_registry = failure_registry
        self.max_concurrency = max(1, max_concurrency)

    def execute(
        self,
        tool_calls: Iterable[ToolCallRequest],
        handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
    ) -> List[ToolCallExecutionResult]:
        calls = list(tool_calls)
        if not calls:
            return []
        results: List[ToolCallExecutionResult] = []
        order = {call.id: index for index, call in enumerate(calls)}
        executor = ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(calls)))
        futures = {}
        cancel_events: Dict[Any, threading.Event] = {}
        try:
            for call in calls:
                context = normalize_tool_context(call.name, call.args, service_name=infer_tool_kind(call.name))
                idempotency_key = tool_idempotency_key(call.name, call.id, context)
                blocked = self.failure_registry.should_block(call.name, call.args)
                if blocked:
                    recovery = recovery_action_for(call.name, "CIRCUIT_OPEN", infer_tool_kind(call.name))
                    details = tool_error_details(call.name, call.args, "CIRCUIT_OPEN", blocked.reason, context=context, call_id=call.id)
                    results.append(
                        ToolCallExecutionResult(
                            id=call.id,
                            name=call.name,
                            status="blocked",
                            idempotency_key=idempotency_key,
                            params_hash=context["paramsHash"],
                            error_type="CIRCUIT_OPEN",
                            error_message=blocked.reason,
                            service_name=infer_tool_kind(call.name),
                            circuit_key=blocked.circuit_key,
                            retryable=False,
                            recommended_action=recovery.action,
                            fallback_tools=list(recovery.fallback_tools),
                            details=details,
                            tool_message=structured_tool_message(call.name, "blocked", "CIRCUIT_OPEN", blocked.reason, recovery, blocked.circuit_key, tool_call_id=call.id, details=details),
                        )
                    )
                    continue
                handler = handlers.get(call.name)
                if handler is None:
                    recovery = recovery_action_for(call.name, "UNKNOWN_TOOL", infer_tool_kind(call.name))
                    details = tool_error_details(call.name, call.args, "UNKNOWN_TOOL", "No handler registered", context=context, call_id=call.id)
                    results.append(
                        ToolCallExecutionResult(
                            id=call.id,
                            name=call.name,
                            status="failed",
                            idempotency_key=idempotency_key,
                            params_hash=context["paramsHash"],
                            error_type="UNKNOWN_TOOL",
                            error_message="No handler registered",
                            service_name=infer_tool_kind(call.name),
                            retryable=False,
                            recommended_action=recovery.action,
                            fallback_tools=list(recovery.fallback_tools),
                            details=details,
                            tool_message=structured_tool_message(call.name, "failed", "UNKNOWN_TOOL", "No handler registered", recovery, tool_call_id=call.id, details=details),
                        )
                    )
                    continue
                cancel_event = threading.Event()
                future = submit_with_current_context(executor, self._run_one, call, handler, cancel_event)
                futures[future] = call
                cancel_events[future] = cancel_event
            if futures:
                max_timeout = max(tool_policy_budget_seconds(self.policy_registry.policy_for(call.name)) for call in futures.values())
                batches = (len(futures) + self.max_concurrency - 1) // self.max_concurrency
                batch_timeout = max_timeout * max(1, batches)
                completed = set()
                try:
                    for future in as_completed(futures, timeout=batch_timeout):
                        completed.add(future)
                        call = futures[future]
                        try:
                            results.append(future.result(timeout=0))
                        except Exception as exc:
                            self.failure_registry.record_failure(call.name, call.args, "ERROR", str(exc))
                            recovery = recovery_action_for(call.name, "ERROR", infer_tool_kind(call.name))
                            error_context = normalize_tool_context(call.name, call.args, service_name=infer_tool_kind(call.name))
                            details = tool_error_details(call.name, call.args, "ERROR", str(exc)[:500], context=error_context, call_id=call.id)
                            results.append(
                                ToolCallExecutionResult(
                                    id=call.id,
                                    name=call.name,
                                    status="failed",
                                    idempotency_key=tool_idempotency_key(call.name, call.id, error_context),
                                    params_hash=error_context["paramsHash"],
                                    error_type="ERROR",
                                    error_message=str(exc)[:500],
                                    service_name=infer_tool_kind(call.name),
                                    retryable=recovery.retryable,
                                    recommended_action=recovery.action,
                                    fallback_tools=list(recovery.fallback_tools),
                                    details=details,
                                    tool_message=structured_tool_message(call.name, "failed", "ERROR", str(exc)[:500], recovery, tool_call_id=call.id, details=details),
                                )
                            )
                except TimeoutError:
                    pass
                for future, call in futures.items():
                    if future in completed:
                        continue
                    cancel_events[future].set()
                    future.cancel()
                    self.failure_registry.record_failure(call.name, call.args, "TIMEOUT", "tool execution timed out")
                    recovery = recovery_action_for(call.name, "TIMEOUT", infer_tool_kind(call.name))
                    timeout_context = normalize_tool_context(call.name, call.args, service_name=infer_tool_kind(call.name))
                    timeout_type = classify_timeout_type("tool execution timed out", infer_tool_kind(call.name))
                    details = tool_error_details(call.name, call.args, "TIMEOUT", "tool execution timed out", timeout_type=timeout_type, context=timeout_context, call_id=call.id)
                    results.append(
                        ToolCallExecutionResult(
                            id=call.id,
                            name=call.name,
                            status="failed",
                            idempotency_key=tool_idempotency_key(call.name, call.id, timeout_context),
                            params_hash=timeout_context["paramsHash"],
                            error_type="TIMEOUT",
                            error_message="tool execution timed out",
                            timeout_type=timeout_type,
                            service_name=infer_tool_kind(call.name),
                            retryable=recovery.retryable,
                            recommended_action=recovery.action,
                            fallback_tools=list(recovery.fallback_tools),
                            details=details,
                            tool_message=structured_tool_message(call.name, "failed", "TIMEOUT", "tool execution timed out", recovery, tool_call_id=call.id, details=details),
                        )
                    )
        finally:
            for future, cancel_event in cancel_events.items():
                if not future.done():
                    cancel_event.set()
                    future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        return sorted(results, key=lambda item: order.get(item.id, 999))

    def _run_one(
        self,
        call: ToolCallRequest,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        batch_cancel_event: threading.Event,
    ) -> ToolCallExecutionResult:
        started = time.monotonic()
        policy = self.policy_registry.policy_for(call.name)
        context = normalize_tool_context(call.name, call.args, service_name=infer_tool_kind(call.name))
        idempotency_key = tool_idempotency_key(call.name, call.id, context)
        attempts = max(1, policy.max_retries + 1)
        last_error = ""
        last_error_type = "ERROR"
        last_timeout_type = ""
        for attempt in range(attempts):
            if batch_cancel_event.is_set():
                last_error = "tool execution canceled after batch timeout"
                last_error_type = "TIMEOUT"
                last_timeout_type = "tool_timeout"
                break
            try:
                result = self._call_with_timeout(
                    handler,
                    call.args,
                    policy.timeout_seconds,
                    batch_cancel_event=batch_cancel_event,
                )
                self.failure_registry.record_success(call.name, call.args)
                contract = validate_tool_result_contract(call.name, result or {}, canonical_tool_registry())
                if not contract.get("valid", True) and tool_contract_enforced(contract):
                    raise RuntimeError("TOOL_CONTRACT_VIOLATION missing=%s" % ",".join(contract.get("missingKeys") or []))
                return ToolCallExecutionResult(
                    id=call.id,
                    name=call.name,
                    status="success",
                    result=result or {},
                    idempotency_key=idempotency_key,
                    params_hash=context["paramsHash"],
                    duration_ms=int((time.monotonic() - started) * 1000),
                    service_name=infer_tool_kind(call.name),
                    contract=contract,
                    result_hash=str(contract.get("resultHash") or ""),
                    tool_message={"toolName": call.name, "status": "success"},
                )
            except Exception as exc:
                last_error = str(exc)
                last_error_type = classify_tool_error(exc)
                last_timeout_type = classify_timeout_type(exc, infer_tool_kind(call.name)) if last_error_type == "TIMEOUT" else ""
                if last_error_type in policy.non_retryable_errors or attempt >= attempts - 1:
                    break
                if policy.retryable_errors and last_error_type not in policy.retryable_errors:
                    break
                if policy.backoff_seconds > 0:
                    if batch_cancel_event.wait(policy.backoff_seconds * (attempt + 1)):
                        last_error = "tool execution canceled after batch timeout"
                        last_error_type = "TIMEOUT"
                        last_timeout_type = "tool_timeout"
                        break
        if not batch_cancel_event.is_set():
            self.failure_registry.record_failure(call.name, call.args, last_error_type, last_error)
        recovery = recovery_action_for(call.name, last_error_type, infer_tool_kind(call.name))
        details = tool_error_details(call.name, call.args, last_error_type, last_error[:500], timeout_type=last_timeout_type, context=context, call_id=call.id)
        return ToolCallExecutionResult(
            id=call.id,
            name=call.name,
            status="failed",
            idempotency_key=idempotency_key,
            params_hash=context["paramsHash"],
            error_type=last_error_type,
            error_message=last_error[:500],
            timeout_type=last_timeout_type,
            duration_ms=int((time.monotonic() - started) * 1000),
            service_name=infer_tool_kind(call.name),
            retryable=recovery.retryable,
            recommended_action=recovery.action,
            fallback_tools=list(recovery.fallback_tools),
            details=details,
            tool_message=structured_tool_message(call.name, "failed", last_error_type, last_error[:500], recovery, tool_call_id=call.id, details=details),
        )

    def _call_with_timeout(
        self,
        handler: Callable[[Dict[str, Any]], Dict[str, Any]],
        args: Dict[str, Any],
        timeout_seconds: int,
        batch_cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        execution_cancel_event = threading.Event()
        context = copy_context()

        def invoke_handler() -> None:
            try:
                with tool_cancel_scope(execution_cancel_event):
                    result_queue.put(("ok", handler(args)))
            except Exception as exc:
                result_queue.put(("error", exc))

        thread = threading.Thread(target=context.run, args=(invoke_handler,), name="tool-call", daemon=True)
        thread.start()
        timeout = max(1, int(timeout_seconds or 1))
        deadline = time.monotonic() + timeout
        while True:
            if batch_cancel_event is not None and batch_cancel_event.is_set():
                execution_cancel_event.set()
                raise TimeoutError("tool execution canceled after batch timeout")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                execution_cancel_event.set()
                raise TimeoutError("tool execution timed out")
            try:
                status, value = result_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                continue
            if status == "error":
                raise value
            return value


def tool_policy_budget_seconds(policy: ToolRuntimePolicy) -> int:
    attempts = max(1, int(policy.max_retries or 0) + 1)
    timeout = max(1, int(policy.timeout_seconds or 1))
    backoff = max(0.0, float(policy.backoff_seconds or 0.0))
    return max(1, int(attempts * timeout + backoff * sum(range(1, attempts)) + 1))


def submit_with_current_context(executor: ThreadPoolExecutor, fn: Any, *args: Any):
    context = copy_context()
    return executor.submit(context.run, fn, *args)
