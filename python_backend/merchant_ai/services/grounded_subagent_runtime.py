from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class IsolatedSubagentJob:
    job_id: str
    thread_id: str
    system_prompt: str
    user_payload: dict[str, Any]
    backend: Any
    tools: list[Any] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    middleware: list[Any] = field(default_factory=list)
    permissions: list[Any] = field(default_factory=list)
    subagents: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class IsolatedSubagentResult:
    job_id: str
    thread_id: str
    checkpoint: dict[str, Any]
    raw_output: str
    update_count: int


class IsolatedSubagentRuntime:
    """Generic isolation Harness used by Skills and other long sub-processes.

    It deliberately knows nothing about Skill semantics, SQL, Topic routing, or
    answer composition.  Callers mount the resources and tools needed by one
    job.  This keeps subagent runtime separate from agent/skill definitions.
    """

    def __init__(
        self,
        *,
        model: Any,
        agent_factory: Any,
        checkpointer: Any,
        checkpoint_config_factory: Optional[
            Callable[[str, str], dict[str, Any]]
        ] = None,
    ) -> None:
        self.model = model
        self.agent_factory = agent_factory
        self.checkpointer = checkpointer
        self.checkpoint_config_factory = checkpoint_config_factory

    def run(
        self,
        job: IsolatedSubagentJob,
        *,
        on_progress: Optional[Callable[[str, str, str], None]] = None,
    ) -> IsolatedSubagentResult:
        if self.checkpointer is None:
            raise RuntimeError("isolated subagent requires a checkpoint backend")
        graph = self.agent_factory(
            model=self.model,
            tools=list(job.tools),
            system_prompt=job.system_prompt,
            middleware=list(job.middleware),
            subagents=list(job.subagents),
            skills=list(job.skills) or None,
            permissions=list(job.permissions),
            backend=job.backend,
            checkpointer=self.checkpointer,
            name="isolated_worker_%s"
            % (re.sub(r"[^a-zA-Z0-9_]+", "_", job.job_id)[:48] or "job"),
        )
        config = (
            self.checkpoint_config_factory(job.thread_id, job.job_id)
            if self.checkpoint_config_factory is not None
            else {
                "configurable": {
                    "thread_id": job.thread_id,
                    "run_id": job.job_id,
                }
            }
        )
        checkpoint = {
            "threadId": job.thread_id,
            "runId": job.job_id,
            "checkpointNamespace": str(
                (config.get("configurable") or {}).get("checkpoint_ns") or ""
            ),
        }
        if on_progress is not None:
            on_progress("subagent", "started", job.job_id)
        update_count = 0
        for update in graph.stream(
            {"messages": [{"role": "user", "content": _json_text(job.user_payload)}]},
            config=config,
            stream_mode="updates",
        ):
            update_count += 1
            if on_progress is not None and update_count <= 32:
                detail = (
                    ",".join(str(key) for key in update.keys())
                    if isinstance(update, dict)
                    else type(update).__name__
                )
                on_progress("subagent_step", "running", detail)
        snapshot = graph.get_state(config)
        messages = list((getattr(snapshot, "values", {}) or {}).get("messages") or [])
        raw_output = _last_assistant_text(messages)
        if on_progress is not None:
            on_progress("subagent", "completed", "updates=%d" % update_count)
        return IsolatedSubagentResult(
            job_id=job.job_id,
            thread_id=job.thread_id,
            checkpoint=checkpoint,
            raw_output=raw_output,
            update_count=update_count,
        )


def _json_text(payload: dict[str, Any]) -> str:
    import json

    return json.dumps(payload, ensure_ascii=False, default=str)


def _last_assistant_text(messages: list[Any]) -> str:
    for message in reversed(messages):
        if str(getattr(message, "type", "") or "") not in {"ai", "assistant"}:
            continue
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                str(item.get("text") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
            return "\n".join(part for part in parts if part).strip()
    return ""
