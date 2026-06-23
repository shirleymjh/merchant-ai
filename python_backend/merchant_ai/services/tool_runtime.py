from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import Any, Callable, Dict, Iterable, List

from merchant_ai.config import Settings
from merchant_ai.models import CircuitBreakerState, ToolCallExecutionResult, ToolCallRequest, ToolFailureRecord, ToolRuntimePolicy


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
                timeout_seconds=max(1, self.settings.agent_node_timeout_seconds),
                max_retries=1,
                backoff_seconds=0.5,
                retryable_errors=["TIMEOUT", "MEM_ALLOC_FAILED", "DORIS_ERROR"],
                non_retryable_errors=["UNKNOWN_COLUMN", "PARSE_ERROR", "UNSAFE_SQL", "UNKNOWN_BASE_TABLE"],
            )
        return ToolRuntimePolicy(tool_name=tool_name, timeout_seconds=max(1, self.settings.agent_node_timeout_seconds), max_retries=0)

    def trace(self) -> List[Dict[str, Any]]:
        names = [
            "inspect_schema",
            "draft_llm_sql",
            "repair_sql",
                "execute_sql",
                "emit_question_understanding",
                "semantic_read",
                "artifact_read",
            ]
        return [self.policy_for(name).model_dump(by_alias=True) for name in names]


class ToolFailureRegistry:
    """Track repeated tool failures and open lightweight circuit breakers."""

    def __init__(self, repeat_threshold: int = 2, circuit_threshold: int = 5):
        self.repeat_threshold = repeat_threshold
        self.circuit_threshold = circuit_threshold
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
        self.circuits[tool_name] = circuit
        return record

    def trace(self) -> Dict[str, Any]:
        return {
            "failures": [item.model_dump(by_alias=True) for item in self.records.values()],
            "circuits": [item.model_dump(by_alias=True) for item in self.circuits.values()],
        }


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
                if "ERROR" in policy.non_retryable_errors or attempt >= attempts - 1:
                    break
                if policy.backoff_seconds > 0:
                    time.sleep(policy.backoff_seconds * (attempt + 1))
        self.failure_registry.record_failure(call.name, call.args, "ERROR", last_error)
        return ToolCallExecutionResult(
            id=call.id,
            name=call.name,
            status="failed",
            error_type="ERROR",
            error_message=last_error[:500],
            duration_ms=int((time.monotonic() - started) * 1000),
        )
