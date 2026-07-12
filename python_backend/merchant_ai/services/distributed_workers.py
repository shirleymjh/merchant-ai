from __future__ import annotations

import json
import multiprocessing
import os
import queue
import socket
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from merchant_ai.config import Settings
from merchant_ai.services.runtime_state import NodeTaskState, RuntimeStateStore, create_runtime_state_store, safe_name


TERMINAL_TASK_STATUSES = {"completed", "failed", "timeout", "canceled"}


class DistributedTaskError(RuntimeError):
    pass


@dataclass
class DistributedTaskResult:
    run_id: str
    task_id: str
    status: str
    result: Dict[str, Any]
    artifact_uri: str = ""
    error: str = ""


class DistributedArtifactStore:
    """Persist cross-process worker payloads on filesystem or S3-compatible storage."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.backend = str(settings.distributed_artifact_backend or "filesystem").lower()
        self.root = settings.resolved_workspace_path / "distributed_artifacts"
        if self.backend == "filesystem":
            self.root.mkdir(parents=True, exist_ok=True)

    def write_json(self, run_id: str, task_id: str, name: str, payload: Dict[str, Any]) -> str:
        content = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        key = self._key(run_id, task_id, name if name.endswith(".json") else "%s.json" % name)
        if self.backend == "s3":
            self._s3_client().put_object(
                Bucket=self._bucket(),
                Key=key,
                Body=content,
                ContentType="application/json",
            )
            return "s3://%s/%s" % (self._bucket(), key)
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return str(path)

    def read_json(self, uri: str) -> Dict[str, Any]:
        if str(uri).startswith("s3://"):
            bucket, _, key = str(uri)[5:].partition("/")
            response = self._s3_client().get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        else:
            content = Path(uri).read_bytes()
        payload = json.loads(content.decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def _key(self, run_id: str, task_id: str, name: str) -> str:
        prefix = str(self.settings.distributed_artifact_s3_prefix or "merchant-ai").strip("/")
        return "/".join([prefix, safe_name(run_id), safe_name(task_id), safe_name(name)])

    def _bucket(self) -> str:
        bucket = str(self.settings.distributed_artifact_s3_bucket or "").strip()
        if not bucket:
            raise DistributedTaskError("YSHOPPING_DISTRIBUTED_ARTIFACT_S3_BUCKET is required")
        return bucket

    def _s3_client(self):
        try:
            import boto3
        except ImportError as exc:
            raise DistributedTaskError("boto3 is required for the s3 artifact backend") from exc
        kwargs = {}
        if self.settings.distributed_artifact_s3_endpoint:
            kwargs["endpoint_url"] = self.settings.distributed_artifact_s3_endpoint
        return boto3.client("s3", **kwargs)


class DistributedSubAgentClient:
    def __init__(
        self,
        settings: Settings,
        state_store: Optional[RuntimeStateStore] = None,
        artifact_store: Optional[DistributedArtifactStore] = None,
    ):
        self.settings = settings
        self.state_store = state_store or create_runtime_state_store(settings)
        self.artifact_store = artifact_store or DistributedArtifactStore(settings)

    def submit(
        self,
        run_id: str,
        task_id: str,
        task_kind: str,
        request: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> NodeTaskState:
        request_uri = self.artifact_store.write_json(run_id, task_id, "request", request)
        state = NodeTaskState(
            run_id=run_id,
            task_id=task_id,
            status="queued",
            idempotency_key="subagent:%s:%s:%s" % (safe_name(task_kind), safe_name(run_id), safe_name(task_id)),
            payload={
                "taskKind": str(task_kind),
                "requestArtifactUri": request_uri,
                "timeoutSeconds": int(timeout_seconds or self.settings.distributed_worker_result_timeout_seconds),
                "submittedAt": time.time(),
            },
        )
        return self.state_store.enqueue_node_task(state)

    def wait(self, run_id: str, task_id: str, timeout_seconds: Optional[int] = None) -> DistributedTaskResult:
        timeout = max(1, int(timeout_seconds or self.settings.distributed_worker_result_timeout_seconds))
        deadline = time.monotonic() + timeout
        poll = max(0.05, float(self.settings.distributed_worker_poll_seconds or 0.5))
        while time.monotonic() < deadline:
            if self.state_store.run_canceled(run_id):
                return DistributedTaskResult(run_id, task_id, "canceled", {}, error="run canceled")
            state = self.state_store.get_node_task(run_id, task_id)
            if state and state.status in TERMINAL_TASK_STATUSES:
                return self._result_from_state(state)
            time.sleep(poll)
        self.state_store.complete_node_task(run_id, task_id, "timeout", {"error": "distributed worker result timeout"})
        return DistributedTaskResult(run_id, task_id, "timeout", {}, error="distributed worker result timeout")

    def execute(
        self,
        run_id: str,
        task_id: str,
        task_kind: str,
        request: Dict[str, Any],
        timeout_seconds: Optional[int] = None,
    ) -> DistributedTaskResult:
        self.submit(run_id, task_id, task_kind, request, timeout_seconds)
        return self.wait(run_id, task_id, timeout_seconds)

    def cancel_run(self, run_id: str, reason: str = "client cancellation") -> None:
        self.state_store.cancel_run(run_id, reason)

    def _result_from_state(self, state: NodeTaskState) -> DistributedTaskResult:
        artifact_uri = str(state.payload.get("resultArtifactUri") or "")
        result = self.artifact_store.read_json(artifact_uri) if artifact_uri else dict(state.payload.get("result") or {})
        return DistributedTaskResult(
            run_id=state.run_id,
            task_id=state.task_id,
            status=state.status,
            result=result,
            artifact_uri=artifact_uri,
            error=str(state.payload.get("error") or ""),
        )


TaskHandler = Callable[[Dict[str, Any], Callable[[], bool]], Dict[str, Any]]


class CancellationProbe:
    def __init__(self, probe: Callable[[], bool]):
        self.probe = probe

    def is_set(self) -> bool:
        return bool(self.probe())


class DistributedSubAgentWorker:
    def __init__(
        self,
        settings: Settings,
        handlers: Optional[Dict[str, TaskHandler]] = None,
        state_store: Optional[RuntimeStateStore] = None,
        artifact_store: Optional[DistributedArtifactStore] = None,
        worker_id: str = "",
    ):
        self.settings = settings
        self.state_store = state_store or create_runtime_state_store(settings)
        self.artifact_store = artifact_store or DistributedArtifactStore(settings)
        self.handlers = dict(handlers or {})
        self.worker_id = worker_id or "%s:%s:%s" % (socket.gethostname(), os.getpid(), uuid.uuid4().hex[:8])
        self._stop = threading.Event()

    def register(self, task_kind: str, handler: TaskHandler) -> None:
        self.handlers[str(task_kind)] = handler

    def run_once(self, task_kinds: Optional[Iterable[str]] = None) -> bool:
        self.state_store.recover_expired_node_tasks(max_attempts=self.settings.distributed_worker_max_attempts)
        state = self.state_store.claim_next_node_task(
            self.worker_id,
            lease_seconds=self.settings.distributed_worker_lease_seconds,
            task_kinds=list(task_kinds or self.handlers.keys()),
        )
        if not state:
            return False
        self._execute_claimed(state)
        return True

    def run_forever(self, task_kinds: Optional[Iterable[str]] = None) -> None:
        poll = max(0.05, float(self.settings.distributed_worker_poll_seconds or 0.5))
        while not self._stop.is_set():
            if not self.run_once(task_kinds):
                self._stop.wait(poll)

    def stop(self) -> None:
        self._stop.set()

    def _execute_claimed(self, state: NodeTaskState) -> None:
        task_kind = str(state.payload.get("taskKind") or "")
        handler = self.handlers.get(task_kind)
        if not handler:
            self.state_store.complete_node_task(state.run_id, state.task_id, "failed", {"error": "unsupported task kind: %s" % task_kind})
            return
        if self.state_store.run_canceled(state.run_id):
            self.state_store.complete_node_task(state.run_id, state.task_id, "canceled", {"error": "run canceled before execution"})
            return
        request = self.artifact_store.read_json(str(state.payload.get("requestArtifactUri") or ""))
        heartbeat_stop = threading.Event()
        heartbeat = threading.Thread(target=self._heartbeat_loop, args=(state, heartbeat_stop), daemon=True)
        heartbeat.start()
        try:
            result = self._run_handler_isolated(state, handler, request)
            if self.state_store.run_canceled(state.run_id):
                self.state_store.complete_node_task(state.run_id, state.task_id, "canceled", {"error": "run canceled during execution"})
                return
            result_uri = self.artifact_store.write_json(state.run_id, state.task_id, "result", result)
            self.state_store.complete_node_task(
                state.run_id,
                state.task_id,
                "completed",
                {"resultArtifactUri": result_uri, "workerId": self.worker_id, "taskKind": task_kind},
            )
        except TimeoutError as exc:
            self.state_store.complete_node_task(
                state.run_id,
                state.task_id,
                "timeout",
                {"error": str(exc)[:1000], "workerId": self.worker_id, "taskKind": task_kind},
            )
        except Exception as exc:
            self.state_store.complete_node_task(
                state.run_id,
                state.task_id,
                "failed",
                {"error": "%s: %s" % (type(exc).__name__, str(exc)[:1000]), "workerId": self.worker_id, "taskKind": task_kind},
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1)

    def _run_handler_isolated(self, state: NodeTaskState, handler: TaskHandler, request: Dict[str, Any]) -> Dict[str, Any]:
        backend = str(self.settings.distributed_worker_execution_backend or "process").lower()

        def canceled() -> bool:
            return self._task_canceled(state)

        if backend in {"inline", "thread"}:
            return handler(request, canceled)
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork" if "fork" in methods else methods[0])
        result_queue = context.Queue(maxsize=1)
        process = context.Process(target=_process_handler_entry, args=(handler, request, result_queue), daemon=True)
        process.start()
        timeout = max(1, int(state.payload.get("timeoutSeconds") or self.settings.distributed_worker_result_timeout_seconds))
        deadline = time.monotonic() + timeout
        try:
            while process.is_alive():
                if canceled():
                    process.terminate()
                    process.join(timeout=2)
                    raise DistributedTaskError("task canceled; worker process terminated")
                if time.monotonic() >= deadline:
                    process.terminate()
                    process.join(timeout=2)
                    raise TimeoutError("task exceeded %s seconds; worker process terminated" % timeout)
                process.join(timeout=min(0.2, max(0.01, deadline - time.monotonic())))
            try:
                status, payload = result_queue.get(timeout=1)
            except queue.Empty as exc:
                raise DistributedTaskError("worker process exited without a result") from exc
            if status == "error":
                raise DistributedTaskError(str(payload))
            return payload if isinstance(payload, dict) else {"value": payload}
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=1)
            result_queue.close()

    def _task_canceled(self, state: NodeTaskState) -> bool:
        if self._stop.is_set() or self.state_store.run_canceled(state.run_id):
            return True
        current = self.state_store.get_node_task(state.run_id, state.task_id)
        return bool(current and current.status in {"canceled", "timeout"})

    def _heartbeat_loop(self, state: NodeTaskState, stop: threading.Event) -> None:
        interval = max(0.2, min(float(self.settings.tool_heartbeat_interval_seconds or 5), self.settings.distributed_worker_lease_seconds / 3))
        while not stop.wait(interval):
            if not self.state_store.heartbeat_node_task(
                state.run_id,
                state.task_id,
                self.worker_id,
                lease_seconds=self.settings.distributed_worker_lease_seconds,
            ):
                return


def builtin_worker_handlers(settings: Settings) -> Dict[str, TaskHandler]:
    return {
        "query_node": lambda request, canceled: execute_query_node_task(settings, request, canceled),
        "analysis_skill": lambda request, canceled: execute_analysis_skill_task(settings, request, canceled),
        "hypothesis_review": lambda request, canceled: execute_hypothesis_review_task(request, canceled),
        "document_analysis": lambda request, canceled: execute_document_analysis_task(settings, request, canceled),
        "python_batch": lambda request, canceled: execute_python_batch_task(settings, request, canceled),
    }


def execute_query_node_task(settings: Settings, request: Dict[str, Any], canceled: Callable[[], bool]) -> Dict[str, Any]:
    from merchant_ai.models import NodeExecutionContext, PlanningAssetPack, QuestionIntent
    from merchant_ai.services.assets import SemanticCatalogService, TopicAssetService
    from merchant_ai.services.llm import LlmClient
    from merchant_ai.services.query import NodeWorkerExecutor, SqlValidationService
    from merchant_ai.services.repositories import DorisRepository

    if canceled():
        raise DistributedTaskError("query node canceled before start")
    worker = NodeWorkerExecutor(
        LlmClient(settings),
        DorisRepository(settings),
        SqlValidationService(),
        settings,
        semantic_catalog=SemanticCatalogService(TopicAssetService(settings)),
    )
    context = NodeExecutionContext.model_validate(request.get("context") or {})
    context.cancel_event = CancellationProbe(canceled)
    result = worker.execute_node(
        QuestionIntent.model_validate(request.get("intent") or {}),
        PlanningAssetPack.model_validate(request.get("assetPack") or {}),
        str(request.get("knowledgeContext") or ""),
        context,
    )
    return result.model_dump(by_alias=True)


def execute_analysis_skill_task(settings: Settings, request: Dict[str, Any], canceled: Callable[[], bool]) -> Dict[str, Any]:
    from merchant_ai.models import AgentRunResult, MerchantInfo, QueryPlan
    from merchant_ai.services.llm import LlmClient
    from merchant_ai.services.skill_worker import SkillWorkerExecutor

    if canceled():
        raise DistributedTaskError("analysis skill canceled before start")
    local_settings = settings.model_copy(update={"distributed_subagents_enabled": False})
    result = SkillWorkerExecutor(LlmClient(local_settings)).execute_answer_skill(
        str(request.get("question") or ""),
        QueryPlan.model_validate(request.get("plan") or {}),
        AgentRunResult.model_validate(request.get("runResult") or {}),
        str(request.get("outputsPath") or ""),
        str(request.get("ruleContext") or ""),
        str(request.get("skillName") or ""),
        merchant=MerchantInfo.model_validate(request.get("merchant") or {}),
        personalization_context=dict(request.get("personalizationContext") or {}),
        initial_trace=dict(request.get("initialTrace") or {}),
    )
    return {"answer": result.answer, "trace": result.trace}


def execute_hypothesis_review_task(request: Dict[str, Any], canceled: Callable[[], bool]) -> Dict[str, Any]:
    from merchant_ai.models import AgentRunResult
    from merchant_ai.services.controlled_react import ControlledReactExplorer

    if canceled():
        raise DistributedTaskError("hypothesis review canceled before start")
    reviews = ControlledReactExplorer().run_parallel_evidence_reviews(
        dict(request.get("hypotheses") or {}),
        AgentRunResult.model_validate(request.get("runResult") or {}),
    )
    return {"reviews": reviews}


def execute_document_analysis_task(settings: Settings, request: Dict[str, Any], canceled: Callable[[], bool]) -> Dict[str, Any]:
    from merchant_ai.services.llm import LlmClient

    if canceled():
        raise DistributedTaskError("document analysis canceled before start")
    content = str(request.get("content") or "")
    question = str(request.get("question") or "请总结文档中的关键事实、风险与待确认项")
    if not content:
        return {"answer": "", "error": "empty document content"}
    answer = LlmClient(settings).chat(
        "你是隔离的文档分析 Sub-Agent。只基于文档内容回答，明确区分事实与推断。",
        "问题：%s\n\n文档：\n%s" % (question, content[:100_000]),
        fallback="",
        timeout_seconds=settings.llm_analysis_timeout_seconds,
    )
    return {"answer": answer, "sourceChars": len(content), "truncated": len(content) > 100_000}


def execute_python_batch_task(settings: Settings, request: Dict[str, Any], canceled: Callable[[], bool]) -> Dict[str, Any]:
    from merchant_ai.services.sandbox import MerchantAnalysisSandbox

    if canceled():
        raise DistributedTaskError("python batch canceled before start")
    script = Path(str(request.get("scriptPath") or ""))
    workspace = Path(str(request.get("workspacePath") or settings.resolved_workspace_path / "python_batch"))
    args = [str(item) for item in request.get("args") or []]
    result = MerchantAnalysisSandbox(settings).run_python(
        script,
        args,
        workspace,
        timeout_seconds=max(1, int(request.get("timeoutSeconds") or settings.skill_worker_timeout_seconds)),
    )
    return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def _process_handler_entry(handler: TaskHandler, request: Dict[str, Any], result_queue: Any) -> None:
    try:
        result_queue.put(("ok", handler(request, lambda: False)))
    except Exception as exc:
        result_queue.put(("error", "%s: %s" % (type(exc).__name__, str(exc)[:2000])))
