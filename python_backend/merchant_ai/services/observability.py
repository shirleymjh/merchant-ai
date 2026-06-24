from __future__ import annotations

import hashlib
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from merchant_ai.models import ArtifactRef, RunStep, TraceSpan


PHASE_BY_NODE = {
    "INHERIT_CONTEXT": "routing",
    "LANGGRAPH_RUNTIME": "routing",
    "MAIN_AGENT_POLICY": "routing",
    "ROUTE_TOPIC": "routing",
    "RETRIEVE_KNOWLEDGE": "retrieval",
    "COMPACT_ASSETS": "retrieval",
    "PLAN_QUERY_GRAPH": "planning",
    "REFLECT_QUERY_GRAPH": "reflection",
    "VALIDATE_QUERY_GRAPH": "validation",
    "REPAIR_QUERY_GRAPH": "planning",
    "EXECUTE_QUERY_GRAPH": "execution",
    "REPAIR_SQL": "execution",
    "VERIFY_EVIDENCE_GRAPH": "evidence",
    "ANSWER_ANALYSIS": "answer",
    "CACHE_ANSWER": "answer",
}


def now_ms() -> float:
    return time.perf_counter() * 1000


def duration_ms(started_ms: float) -> int:
    return max(0, int(now_ms() - started_ms))


def text_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:16]


def artifact_ref_from_path(path: str, namespace: str = "", reason: str = "") -> ArtifactRef:
    if not path:
        return ArtifactRef(namespace=namespace, reason=reason)
    file_path = Path(path)
    size = 0
    digest = ""
    try:
        if file_path.exists() and file_path.is_file():
            data = file_path.read_bytes()
            size = len(data)
            digest = hashlib.sha256(data).hexdigest()
    except Exception:
        size = 0
        digest = ""
    return ArtifactRef(
        artifact_id="artifact_" + uuid.uuid4().hex,
        namespace=namespace,
        path=str(file_path),
        relative_path=file_path.name,
        title=file_path.name,
        reason=reason,
        bytes=size,
        estimated_chars=size,
        sha256=digest,
    )


def start_step(state: Dict[str, Any], action_id: str, agent: str, node: str, reason: str = "", input_summary: str = "") -> RunStep:
    step = RunStep(
        step_id="step_" + uuid.uuid4().hex,
        run_id=str(state.get("run_id") or ""),
        action_id=action_id,
        agent=agent,
        node=node,
        status="running",
        reason=reason,
        input_summary=input_summary[:2000],
    )
    state.setdefault("run_steps", []).append(step)
    state["_active_step_id"] = step.step_id
    return step


def finish_step(
    state: Dict[str, Any],
    step: RunStep,
    status: str,
    output_summary: str = "",
    error_code: str = "",
    error_message: str = "",
    artifact_refs: Optional[List[ArtifactRef]] = None,
) -> RunStep:
    step.status = status
    step.end_time = datetime.now()
    step.duration_ms = max(0, int((step.end_time - step.start_time).total_seconds() * 1000))
    step.output_summary = output_summary[:2000]
    step.error_code = error_code
    step.error_message = error_message[:2000]
    step.artifact_refs = artifact_refs or []
    state["_active_step_id"] = ""
    return step


def append_span(
    state: Dict[str, Any],
    kind: str,
    name: str,
    started_ms: float,
    status: str = "success",
    step_id: str = "",
    model: str = "",
    provider: str = "",
    estimated_prompt_chars: int = 0,
    estimated_completion_chars: int = 0,
    sql: str = "",
    table: str = "",
    row_count: int = 0,
    error_code: str = "",
    error_message: str = "",
    retry_or_fallback_count: int = 0,
    artifact_refs: Optional[List[ArtifactRef]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> TraceSpan:
    span = TraceSpan(
        span_id="span_" + uuid.uuid4().hex,
        run_id=str(state.get("run_id") or ""),
        step_id=step_id or str(state.get("_active_step_id") or ""),
        kind=kind,
        name=name,
        status=status,
        end_time=datetime.now(),
        duration_ms=duration_ms(started_ms),
        model=model,
        provider=provider,
        estimated_prompt_chars=estimated_prompt_chars,
        estimated_completion_chars=estimated_completion_chars,
        sql_hash=text_hash(sql) if sql else "",
        table=table,
        row_count=row_count,
        error_code=error_code,
        error_message=error_message[:2000],
        retry_or_fallback_count=retry_or_fallback_count,
        artifact_refs=artifact_refs or [],
        metadata=metadata or {},
    )
    state.setdefault("trace_spans", []).append(span)
    return span


def normalize_model_dump(items: Iterable[Any]) -> List[Dict[str, Any]]:
    dumped: List[Dict[str, Any]] = []
    for item in items:
        if hasattr(item, "model_dump"):
            dumped.append(item.model_dump(by_alias=True))
        elif isinstance(item, dict):
            dumped.append(item)
    return dumped


def performance_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    spans = [item for item in state.get("trace_spans", []) if hasattr(item, "kind")]
    steps = [item for item in state.get("run_steps", []) if hasattr(item, "node")]
    phases: Dict[str, Dict[str, Any]] = {}
    for step in steps:
        phase = PHASE_BY_NODE.get(step.node, "other")
        item = phases.setdefault(phase, {"durationMs": 0, "steps": 0, "errors": 0})
        item["durationMs"] += int(step.duration_ms or 0)
        item["steps"] += 1
        if step.status not in {"success", "completed", "skipped"}:
            item["errors"] += 1
    llm_spans = [span for span in spans if span.kind == "llm"]
    sql_spans = [span for span in spans if span.kind in {"sql", "doris_query"}]
    slow_spans = sorted(spans, key=lambda span: int(span.duration_ms or 0), reverse=True)[:8]
    return {
        "totalDurationMs": sum(int(step.duration_ms or 0) for step in steps),
        "phases": phases,
        "spanCount": len(spans),
        "stepCount": len(steps),
        "llm": {
            "count": len(llm_spans),
            "durationMs": sum(int(span.duration_ms or 0) for span in llm_spans),
            "timeouts": sum(1 for span in llm_spans if "TIMEOUT" in span.error_code or "timeout" in span.error_message.lower()),
            "estimatedPromptChars": sum(int(span.estimated_prompt_chars or 0) for span in llm_spans),
            "estimatedCompletionChars": sum(int(span.estimated_completion_chars or 0) for span in llm_spans),
        },
        "sql": {
            "count": len(sql_spans),
            "durationMs": sum(int(span.duration_ms or 0) for span in sql_spans),
            "rows": sum(int(span.row_count or 0) for span in sql_spans),
            "errors": sum(1 for span in sql_spans if span.status != "success"),
        },
        "slowSpans": [
            {
                "name": span.name,
                "kind": span.kind,
                "durationMs": span.duration_ms,
                "status": span.status,
                "errorCode": span.error_code,
            }
            for span in slow_spans
        ],
    }

