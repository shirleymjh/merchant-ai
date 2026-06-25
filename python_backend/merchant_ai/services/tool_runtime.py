from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional
from urllib import request as url_request

from merchant_ai.config import Settings
from merchant_ai.models import (
    CircuitBreakerState,
    LoadBalancerTarget,
    RuntimeAlert,
    ToolCachePolicy,
    ToolCallExecutionResult,
    ToolCallRequest,
    ToolFailureRecord,
    ToolRuntimeMetrics,
    ToolRuntimePolicy,
)


def now_ms() -> int:
    return int(time.time() * 1000)


def classify_tool_error(error: Exception | str) -> str:
    text = str(error or "")
    lower = text.lower()
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
    if "invalid" in lower or "bad argument" in lower:
        return "INVALID_ARGUMENT"
    if "provider" in lower or "llm" in lower or "model" in lower:
        return "PROVIDER_ERROR"
    return "ERROR"


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
            return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=5, max_retries=0, non_retryable_errors=["UNKNOWN_COLUMN", "INVALID_ARGUMENT"])
        if tool_name in {"semantic_ls", "semantic_read", "semantic_grep", "semantic_write", "artifact_ls", "artifact_read", "artifact_grep", "artifact_write"}:
            return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=5, max_retries=0, non_retryable_errors=["INVALID_ARGUMENT", "ARTIFACT_NOT_FOUND", "SEMANTIC_REF_NOT_FOUND"])
        if tool_name in {"draft_llm_sql", "repair_sql", "emit_question_understanding", "draft_sql"}:
            return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=max(1, self.settings.llm_request_timeout_seconds), max_retries=1, backoff_seconds=0.2, retryable_errors=["TIMEOUT", "PROVIDER_ERROR"])
        if tool_name == "execute_sql":
            return ToolRuntimePolicy(
                tool_name=tool_name,
                timeout_seconds=max(1, self.settings.doris_read_timeout_seconds),
                max_retries=0,
                backoff_seconds=0.0,
                retryable_errors=[],
                non_retryable_errors=["UNKNOWN_COLUMN", "MEM_ALLOC_FAILED", "TIMEOUT", "PARSE_ERROR", "UNSAFE_SQL", "UNKNOWN_BASE_TABLE"],
            )
        if tool_name in {"node_agent", "node_agent_batch"}:
            return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=max(1, self.settings.agent_node_timeout_seconds), max_retries=0, non_retryable_errors=["TIMEOUT"])
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

    def fingerprint(self, tool_name: str, args: Any) -> str:
        try:
            payload = json.dumps(args, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            payload = str(args)
        digest = hashlib.sha256(("%s:%s" % (tool_name, payload)).encode("utf-8")).hexdigest()[:16]
        return "%s:%s" % (tool_name, digest)

    def should_block(self, tool_name: str, args: Any) -> CircuitBreakerState | None:
        circuit = self.circuits.get(tool_name)
        if circuit and circuit.open:
            if circuit.open_until_ms and circuit.open_until_ms <= now_ms():
                circuit.open = False
                circuit.failure_count = 0
                circuit.reason = ""
                self.circuits[tool_name] = circuit
            else:
                return circuit
        circuit = self.circuits.get(tool_name)
        if circuit and circuit.open:
            return circuit
        record = self.records.get(self.fingerprint(tool_name, args))
        if record and record.blocked:
            return CircuitBreakerState(tool_name=tool_name, open=True, failure_count=record.count, reason="repeated identical failure: %s" % record.error_type)
        return None

    def record_success(self, tool_name: str, args: Any) -> None:
        self.records.pop(self.fingerprint(tool_name, args), None)
        circuit = self.circuits.get(tool_name)
        if circuit:
            circuit.failure_count = 0
            circuit.open = False
            circuit.reason = ""
            circuit.open_until_ms = 0

    def record_failure(self, tool_name: str, args: Any, error_type: str, error_message: str) -> ToolFailureRecord:
        fingerprint = self.fingerprint(tool_name, args)
        record = self.records.get(fingerprint) or ToolFailureRecord(fingerprint=fingerprint, tool_name=tool_name)
        record.error_type = error_type or "ERROR"
        record.error_message = str(error_message or "")[:500]
        record.count += 1
        record.blocked = record.count >= self.repeat_threshold
        self.records[fingerprint] = record
        circuit = self.circuits.get(tool_name) or CircuitBreakerState(tool_name=tool_name)
        circuit.failure_count += 1
        if circuit.failure_count >= self.circuit_threshold:
            circuit.open = True
            circuit.reason = "tool failure threshold reached: %s" % record.error_type
            circuit.opened_at_ms = now_ms()
            circuit.open_until_ms = circuit.opened_at_ms + self.cooldown_seconds * 1000
        self.circuits[tool_name] = circuit
        return record

    def trace(self) -> Dict[str, Any]:
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
    ):
        self.settings = settings
        self.policy_registry = policy_registry or ToolRuntimePolicyRegistry(settings)
        self.failure_registry = failure_registry or ToolFailureRegistry(
            repeat_threshold=settings.tool_failure_repeat_threshold,
            circuit_threshold=settings.tool_circuit_threshold,
            cooldown_seconds=settings.tool_circuit_cooldown_seconds,
        )
        self.cache_store = cache_store or MemoryCacheStore(settings.cache_memory_max_entries)
        self.rate_limit_store = rate_limit_store or MemoryRateLimitStore()
        self.load_balancer = load_balancer or default_load_balancer(settings)
        self.metrics = metrics or RuntimeMetricsAggregator()
        self.alert_manager = RuntimeAlertManager(settings, self.metrics)

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
    ) -> ToolCallExecutionResult:
        call = ToolCallRequest(id=call_id or tool_name, name=tool_name, args=args or {})
        blocked = self.failure_registry.should_block(tool_name, call.args)
        if blocked:
            result = ToolCallExecutionResult(
                id=call.id,
                name=tool_name,
                status="blocked",
                error_type="CIRCUIT_OPEN",
                error_message=blocked.reason,
            )
            self.metrics.record(tool_name, result.status, 0, result.error_type, circuit_blocked=True)
            self.alert_manager.evaluate()
            return result
        if self.settings.tool_rate_limit_enabled:
            rate_key = target_kind or self._tool_kind(tool_name)
            if not self.rate_limit_store.allow(rate_key, self._tool_qps(tool_name, rate_key)):
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="blocked",
                    error_type="RATE_LIMITED",
                    error_message="tool rate limited: %s" % rate_key,
                    rate_limited=True,
                )
                self.metrics.record(tool_name, result.status, 0, result.error_type, rate_limited=True)
                return result
        cache_key = ""
        if cache_policy and cache_policy.enabled and self.settings.cache_enabled:
            cache_key = self.cache_key(tool_name, call.args, cache_policy)
            cached = self.cache_store.get(cache_key)
            if cached is not None:
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="success",
                    result=cached if isinstance(cached, dict) else {"value": cached},
                    cache_hit=True,
                    cache_key=cache_key,
                    attempts=0,
                )
                self.metrics.record(tool_name, result.status, 0, cache_hit=True)
                return result
        started = time.monotonic()
        policy = self.policy_registry.policy_for(tool_name)
        attempts = max(1, policy.max_retries + 1)
        last_error = ""
        last_error_type = "ERROR"
        selected_target = self.load_balancer.select(target_kind or self._tool_kind(tool_name))
        for attempt in range(attempts):
            try:
                next_args = dict(call.args)
                if selected_target.endpoint:
                    next_args.setdefault("_target", selected_target.model_dump(by_alias=True))
                value = self._call_with_timeout(handler, next_args, policy.timeout_seconds)
                duration_ms = int((time.monotonic() - started) * 1000)
                self.failure_registry.record_success(tool_name, call.args)
                if cache_key and cache_policy:
                    self.cache_store.set(cache_key, value or {}, cache_policy.ttl_seconds or self.settings.semantic_cache_ttl_seconds)
                result = ToolCallExecutionResult(
                    id=call.id,
                    name=tool_name,
                    status="success",
                    result=value or {},
                    duration_ms=duration_ms,
                    attempts=attempt + 1,
                    cache_hit=False,
                    cache_key=cache_key,
                    target=selected_target.name,
                )
                self.metrics.record(tool_name, result.status, duration_ms, attempts=result.attempts, target=selected_target.name)
                self.alert_manager.evaluate()
                return result
            except Exception as exc:
                last_error = str(exc)
                last_error_type = classify_tool_error(exc)
                if last_error_type in policy.non_retryable_errors or attempt >= attempts - 1:
                    break
                if policy.retryable_errors and last_error_type not in policy.retryable_errors:
                    break
                if policy.backoff_seconds > 0:
                    time.sleep(policy.backoff_seconds * (attempt + 1))
        duration_ms = int((time.monotonic() - started) * 1000)
        self.failure_registry.record_failure(tool_name, call.args, last_error_type, last_error)
        result = ToolCallExecutionResult(
            id=call.id,
            name=tool_name,
            status="failed",
            error_type=last_error_type,
            error_message=last_error[:500],
            duration_ms=duration_ms,
            attempts=attempts,
            cache_key=cache_key,
            target=selected_target.name,
        )
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
        with ThreadPoolExecutor(max_workers=min(max(1, self.settings.tool_max_concurrency), len(calls))) as executor:
            futures = {}
            for call in calls:
                handler = handlers.get(call.name)
                if handler is None:
                    results.append(ToolCallExecutionResult(id=call.id, name=call.name, status="failed", error_type="UNKNOWN_TOOL", error_message="No handler registered"))
                    continue
                futures[executor.submit(self.execute, call.name, call.args, handler, call.id, (cache_policies or {}).get(call.name))] = call
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    call = futures[future]
                    results.append(ToolCallExecutionResult(id=call.id, name=call.name, status="failed", error_type=classify_tool_error(exc), error_message=str(exc)[:500]))
        return sorted(results, key=lambda item: order.get(item.id, 999))

    def trace(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics.trace(),
            "rateLimits": self.rate_limit_store.trace(),
            "cache": self.cache_store.trace(),
            "loadBalancer": self.load_balancer.trace(),
            "alerts": self.alert_manager.trace(),
        }

    def _call_with_timeout(self, handler: Callable[[Dict[str, Any]], Dict[str, Any]], args: Dict[str, Any], timeout_seconds: int) -> Dict[str, Any]:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(handler, args)
        try:
            return future.result(timeout=max(1, int(timeout_seconds or 1)))
        except TimeoutError as exc:
            future.cancel()
            raise TimeoutError("tool execution timed out") from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

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
        with ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(calls))) as executor:
            futures = {}
            for call in calls:
                blocked = self.failure_registry.should_block(call.name, call.args)
                if blocked:
                    results.append(
                        ToolCallExecutionResult(
                            id=call.id,
                            name=call.name,
                            status="blocked",
                            error_type="CIRCUIT_OPEN",
                            error_message=blocked.reason,
                        )
                    )
                    continue
                handler = handlers.get(call.name)
                if handler is None:
                    results.append(
                        ToolCallExecutionResult(id=call.id, name=call.name, status="failed", error_type="UNKNOWN_TOOL", error_message="No handler registered")
                    )
                    continue
                futures[executor.submit(self._run_one, call, handler)] = call
            if futures:
                max_timeout = max(max(1, self.policy_registry.policy_for(call.name).timeout_seconds) for call in futures.values())
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
                            results.append(
                                ToolCallExecutionResult(
                                    id=call.id,
                                    name=call.name,
                                    status="failed",
                                    error_type="ERROR",
                                    error_message=str(exc)[:500],
                                )
                            )
                except TimeoutError:
                    pass
                for future, call in futures.items():
                    if future in completed:
                        continue
                    future.cancel()
                    self.failure_registry.record_failure(call.name, call.args, "TIMEOUT", "tool execution timed out")
                    results.append(
                        ToolCallExecutionResult(
                            id=call.id,
                            name=call.name,
                            status="failed",
                            error_type="TIMEOUT",
                            error_message="tool execution timed out",
                        )
                    )
        return sorted(results, key=lambda item: order.get(item.id, 999))

    def _run_one(self, call: ToolCallRequest, handler: Callable[[Dict[str, Any]], Dict[str, Any]]) -> ToolCallExecutionResult:
        started = time.monotonic()
        policy = self.policy_registry.policy_for(call.name)
        attempts = max(1, policy.max_retries + 1)
        last_error = ""
        last_error_type = "ERROR"
        for attempt in range(attempts):
            try:
                result = handler(call.args)
                self.failure_registry.record_success(call.name, call.args)
                return ToolCallExecutionResult(
                    id=call.id,
                    name=call.name,
                    status="success",
                    result=result or {},
                    duration_ms=int((time.monotonic() - started) * 1000),
                )
            except Exception as exc:
                last_error = str(exc)
                last_error_type = classify_tool_error(exc)
                if last_error_type in policy.non_retryable_errors or attempt >= attempts - 1:
                    break
                if policy.retryable_errors and last_error_type not in policy.retryable_errors:
                    break
                if policy.backoff_seconds > 0:
                    time.sleep(policy.backoff_seconds * (attempt + 1))
        self.failure_registry.record_failure(call.name, call.args, last_error_type, last_error)
        return ToolCallExecutionResult(
            id=call.id,
            name=call.name,
            status="failed",
            error_type=last_error_type,
            error_message=last_error[:500],
            duration_ms=int((time.monotonic() - started) * 1000),
        )
