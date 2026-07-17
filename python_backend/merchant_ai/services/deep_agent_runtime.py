"""Diana-style Deep Agent outer harness around the governed domain kernel.

Deep Agents owns the single Core ReAct loop, run context envelope, filesystem,
summarization, skills, subagents, tool-call patching and durable conversation
checkpoint. Existing domain actions remain governed tools: merchant memory,
QueryGraph, tenant, SQL and evidence gates stay authoritative while the Core
model chooses the next safe action.
"""

from __future__ import annotations

import asyncio
import fnmatch
import hashlib
import json
import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterator, List, Optional

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from deepagents.backends.protocol import EditResult, FileInfo, GlobResult, GrepMatch, GrepResult, LsResult, ReadResult, WriteResult
from langchain.tools import ToolRuntime, tool

from merchant_ai.graph.state import AgentState, emit, register_event_listener, unregister_event_listener
from merchant_ai.models import AgentActionTrace, AgentDecision, ChatContext, ChatResponse, ConversationMessage
from merchant_ai.services.assets import build_stable_topic_table_manifest
from merchant_ai.services.context_filesystem import ContextPathOutsideRootError, resolve_context_path
from merchant_ai.services.tool_runtime import tool_runtime_scope


LOGGER = logging.getLogger(__name__)
_KNOWLEDGE_SCOPE: ContextVar[Optional[Any]] = ContextVar("diana_knowledge_scope", default=None)


@dataclass
class _ResultSink:
    response: Optional[ChatResponse] = None


@dataclass
class _DianaLeadSession:
    state: AgentState
    sink: _ResultSink
    available_actions: tuple[str, ...] = ()
    observation: Optional[Dict[str, Any]] = None
    action_count: int = 0
    table_manifest_disclosed: bool = False
    core_semantic_evidence: List[Dict[str, Any]] = field(default_factory=list)
    terminal: bool = False
    lock: Any = field(default_factory=RLock, repr=False)


@dataclass(frozen=True)
class DeepAgentRunContext:
    question: str
    merchant_id: str
    chat_context: Optional[ChatContext]
    listener: Any
    thread_id: str
    run_id: str
    message_history: tuple[ConversationMessage, ...]
    sink: _ResultSink
    # Keep the mutable domain session opaque to Pydantic tool-schema
    # generation; it is trusted runtime context, never a model argument.
    session: Any


class ReadOnlySemanticBackend:
    """Deep Agents backend exposing only the governed semantic virtual tree."""

    MAX_EVIDENCE_ITEMS = 64
    MAX_EVIDENCE_SNIPPET_CHARS = 4_000

    def __init__(self, semantic_catalog: Any):
        self.semantic_catalog = semantic_catalog

    @contextmanager
    def scope_to_state(self, state: AgentState) -> Iterator[None]:
        """Bind every semantic file operation to the active Topic workspace."""

        token = _KNOWLEDGE_SCOPE.set(state)
        try:
            yield
        finally:
            _KNOWLEDGE_SCOPE.reset(token)

    @contextmanager
    def scope_to_session(self, session: _DianaLeadSession) -> Iterator[None]:
        """Track the session so scope follows state replacements between actions."""

        token = _KNOWLEDGE_SCOPE.set(session)
        try:
            yield
        finally:
            _KNOWLEDGE_SCOPE.reset(token)

    @staticmethod
    def _path(value: str) -> str:
        return str(value or "/").strip().lstrip("/").rstrip("/")

    @staticmethod
    def _topic_from_path(path: str) -> str:
        parts = [part for part in str(path or "").strip("/").split("/") if part]
        return parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""

    def _allowed_topics(self) -> List[str]:
        scope = _KNOWLEDGE_SCOPE.get()
        if not scope:
            return []
        state = scope.state if isinstance(scope, _DianaLeadSession) else scope
        workspace = state.get("topic_workspace") or {}
        return list(dict.fromkeys(str(item).strip() for item in workspace.get("topics") or [] if str(item).strip()))

    def _scope_error(self, path: str = "") -> str:
        allowed = self._allowed_topics()
        if not allowed:
            return "TOPIC_SCOPE_REQUIRED: route and confirm a Topic before reading /knowledge"
        topic = self._topic_from_path(path)
        if topic and topic not in allowed:
            return "TOPIC_SCOPE_DENIED: %s is outside the active Topic workspace" % topic
        return ""

    def _record_core_read_evidence(
        self,
        result: Dict[str, Any],
        requested_path: str,
        content: str,
        offset: int,
        limit: int,
    ) -> None:
        """Record only successful Core ``read_file`` results as trusted evidence.

        ``ls`` and ``grep`` are navigation aids and deliberately never enter
        this ledger.  The ref metadata comes from the governed catalog result,
        not from model-authored tool arguments.
        """

        scope = _KNOWLEDGE_SCOPE.get()
        if not isinstance(scope, _DianaLeadSession):
            return
        topic = str(result.get("topic") or "").strip()
        ref_id = str(result.get("refId") or "").strip()
        if not ref_id or topic not in self._allowed_topics():
            return
        excerpt = str(content or "")[: self.MAX_EVIDENCE_SNIPPET_CHARS]
        evidence = {
            "refId": ref_id,
            "path": str(result.get("path") or requested_path).strip().lstrip("/"),
            "kind": str(result.get("kind") or "").strip().upper(),
            "topic": topic,
            "table": str(result.get("table") or "").strip(),
            "contentSnippet": excerpt,
            "contentHash": hashlib.sha256(str(content or "").encode("utf-8")).hexdigest(),
            "offset": max(0, int(offset or 0)),
            "limit": max(1, int(limit or 1)),
        }
        identity = (
            evidence["refId"],
            evidence["path"],
            evidence["offset"],
            evidence["contentHash"],
        )
        with scope.lock:
            retained = [
                item
                for item in scope.core_semantic_evidence
                if (
                    str(item.get("refId") or ""),
                    str(item.get("path") or ""),
                    int(item.get("offset") or 0),
                    str(item.get("contentHash") or ""),
                )
                != identity
            ]
            retained.append(evidence)
            scope.core_semantic_evidence = retained[-self.MAX_EVIDENCE_ITEMS :]

    def ls(self, path: str) -> LsResult:
        normalized = self._path(path)
        if not normalized:
            return LsResult(entries=[FileInfo(path="/topics/", is_dir=True, size=0, modified_at="")])
        if normalized == "topics":
            topics = self._allowed_topics()
            if not topics:
                return LsResult(error=self._scope_error(normalized))
            return LsResult(
                entries=[FileInfo(path="/topics/%s/" % topic, is_dir=True, size=0, modified_at="") for topic in topics]
            )
        scope_error = self._scope_error(normalized)
        if scope_error:
            return LsResult(error=scope_error)
        try:
            refs = self.semantic_catalog.ls(path=normalized, limit=500)
        except Exception as exc:
            return LsResult(error="SEMANTIC_LS_FAILED: %s" % str(exc)[:200])
        entries = [
            FileInfo(
                path="/" + str(item.get("path") or "").lstrip("/"),
                is_dir=False,
                size=int(item.get("estimatedChars") or 0),
                modified_at="",
            )
            for item in refs
            if isinstance(item, dict) and item.get("path")
        ]
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        normalized = self._path(file_path)
        scope_error = self._scope_error(normalized)
        if scope_error:
            return ReadResult(error=scope_error)
        try:
            result = self.semantic_catalog.read(path=normalized, max_chars=2_000_000, offset=0)
        except Exception as exc:
            return ReadResult(error="SEMANTIC_READ_FAILED: %s" % str(exc)[:200])
        if not result.get("success"):
            return ReadResult(error=str(result.get("error") or "SEMANTIC_REF_NOT_FOUND"))
        if str(result.get("topic") or "") not in self._allowed_topics():
            return ReadResult(error="TOPIC_SCOPE_DENIED: semantic ref escaped the active Topic workspace")
        lines = str(result.get("content") or "").splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        content = "".join(lines[start:end])
        self._record_core_read_evidence(result, normalized, content, start, limit)
        return ReadResult(
            file_data={
                "content": content,
                "encoding": "utf-8",
            }
        )

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        normalized = self._path(path or "")
        scope_error = self._scope_error(normalized)
        if scope_error:
            return GrepResult(error=scope_error)
        requested_topic = self._topic_from_path(normalized)
        topics = [requested_topic] if requested_topic else self._allowed_topics()
        hits: List[Dict[str, Any]] = []
        try:
            for topic in topics:
                hits.extend(self.semantic_catalog.grep(query=pattern, topic=topic, limit=100))
        except Exception as exc:
            return GrepResult(error="SEMANTIC_GREP_FAILED: %s" % str(exc)[:200])
        matches: List[GrepMatch] = []
        for hit in hits:
            hit_path = "/" + str(hit.get("path") or "").lstrip("/")
            if glob and not fnmatch.fnmatch(hit_path, glob):
                continue
            snippets = hit.get("snippets") or [hit.get("summary") or hit.get("title") or ""]
            for snippet in snippets[:3]:
                matches.append(GrepMatch(path=hit_path, line=1, text=str(snippet)))
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        listing = self.ls(path or "/")
        if listing.error:
            return GlobResult(error=listing.error)
        return GlobResult(matches=[item for item in listing.entries or [] if fnmatch.fnmatch(item.get("path", ""), pattern)])

    def write(self, file_path: str, content: str) -> WriteResult:
        del content
        return WriteResult(error="SEMANTIC_BACKEND_READ_ONLY", path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        del old_string, new_string, replace_all
        return EditResult(error="SEMANTIC_BACKEND_READ_ONLY", path=file_path)

    def ls_info(self, path: str) -> List[FileInfo]:
        result = self.ls(path)
        return result.entries or []

    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        result = self.glob(pattern, path)
        return result.matches or []

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> List[GrepMatch] | str:
        result = self.grep(pattern, path, glob)
        return result.error or result.matches or []

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self.grep(pattern, path, glob)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.glob(pattern, path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def als_info(self, path: str) -> List[FileInfo]:
        return self.ls_info(path)

    async def aglob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        return self.glob_info(pattern, path)

    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> List[GrepMatch] | str:
        return self.grep_raw(pattern, path, glob)


class ReadOnlyRunArtifactBackend:
    """Expose only the active Diana run's intermediate files to DeepAgent.

    The domain kernel remains the sole writer of QueryGraph, SQL/result and
    evidence artifacts. Mounting that existing store at ``/artifacts`` lets the
    Core progressively inspect large intermediates with native file tools
    without copying them into prompt state or creating a second artifact store.
    """

    MAX_GREP_FILES = 200
    MAX_FILE_CHARS = 2_000_000
    MAX_MATCHES = 100

    def __init__(self, settings: Any):
        self.settings = settings

    @staticmethod
    def _path(value: str) -> str:
        return str(value or "/").strip().lstrip("/").rstrip("/")

    def _active_root(self) -> tuple[Path | None, str]:
        scope = _KNOWLEDGE_SCOPE.get()
        if not isinstance(scope, _DianaLeadSession):
            return None, "ARTIFACT_SCOPE_REQUIRED: artifacts are available only inside the active Diana run"
        state = scope.state
        thread_data = state.get("thread_data")
        outputs_path = str(
            getattr(thread_data, "outputs_path", "")
            or (thread_data.get("outputs_path") if isinstance(thread_data, dict) else "")
            or ""
        )
        if not outputs_path:
            return None, "ARTIFACT_SCOPE_REQUIRED: active run has no outputs workspace"
        workspace_root = Path(self.settings.resolved_workspace_path).resolve()
        candidate = (Path(outputs_path) / "artifacts").resolve(strict=False)
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            return None, "ARTIFACT_SCOPE_DENIED: active run artifact root escaped the workspace"
        if not candidate.exists():
            return None, "ARTIFACT_NOT_READY: this run has not produced intermediate files yet"
        return candidate, ""

    def _resolve(self, path: str) -> tuple[Path | None, Path | None, str]:
        root, error = self._active_root()
        if root is None:
            return None, None, error
        try:
            target = resolve_context_path(root, self._path(path))
        except ContextPathOutsideRootError:
            return root, None, "ARTIFACT_PATH_OUTSIDE_ROOT"
        return root, target, ""

    @staticmethod
    def _file_info(root: Path, path: Path) -> FileInfo:
        return FileInfo(
            path="/" + str(path.relative_to(root)).replace("\\", "/") + ("/" if path.is_dir() else ""),
            is_dir=path.is_dir(),
            size=0 if path.is_dir() else int(path.stat().st_size),
            modified_at="",
        )

    def ls(self, path: str) -> LsResult:
        root, target, error = self._resolve(path)
        if error:
            return LsResult(error=error)
        assert root is not None and target is not None
        if not target.exists():
            return LsResult(error="ARTIFACT_NOT_FOUND")
        if target.is_file():
            return LsResult(entries=[self._file_info(root, target)])
        entries: List[FileInfo] = []
        try:
            for child in sorted(target.iterdir(), key=lambda item: item.name):
                safe_child = resolve_context_path(root, child)
                entries.append(self._file_info(root, safe_child))
        except (ContextPathOutsideRootError, OSError) as exc:
            return LsResult(error="ARTIFACT_LS_FAILED: %s" % str(exc)[:160])
        return LsResult(entries=entries)

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        _, target, error = self._resolve(file_path)
        if error:
            return ReadResult(error=error)
        assert target is not None
        if not target.exists() or not target.is_file():
            return ReadResult(error="ARTIFACT_NOT_FOUND")
        try:
            text = target.read_text(encoding="utf-8")[: self.MAX_FILE_CHARS]
        except (OSError, UnicodeError) as exc:
            return ReadResult(error="ARTIFACT_READ_FAILED: %s" % str(exc)[:160])
        lines = text.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        return ReadResult(file_data={"content": "".join(lines[start:end]), "encoding": "utf-8"})

    def grep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        root, target, error = self._resolve(path or "/")
        if error:
            return GrepResult(error=error)
        assert root is not None and target is not None
        # Artifact search is intentionally literal. Model-authored regular
        # expressions over large one-line JSON files can create pathological
        # backtracking; filename globbing remains available separately.
        needle = str(pattern or "")[:500].casefold()
        candidates = [target] if target.is_file() else list(target.rglob("*"))
        matches: List[GrepMatch] = []
        inspected = 0
        for candidate in candidates:
            if inspected >= self.MAX_GREP_FILES or len(matches) >= self.MAX_MATCHES:
                break
            try:
                safe_candidate = resolve_context_path(root, candidate)
            except ContextPathOutsideRootError:
                continue
            if not safe_candidate.is_file():
                continue
            relative = "/" + str(safe_candidate.relative_to(root)).replace("\\", "/")
            if glob and not fnmatch.fnmatch(relative, glob):
                continue
            inspected += 1
            try:
                lines = safe_candidate.read_text(encoding="utf-8")[: self.MAX_FILE_CHARS].splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if needle in line[:20_000].casefold():
                    matches.append(GrepMatch(path=relative, line=line_number, text=line[:500]))
                    if len(matches) >= self.MAX_MATCHES:
                        break
        return GrepResult(matches=matches)

    def glob(self, pattern: str, path: str | None = None) -> GlobResult:
        root, target, error = self._resolve(path or "/")
        if error:
            return GlobResult(error=error)
        assert root is not None and target is not None
        candidates = [target] if target.is_file() else list(target.rglob("*"))
        matches: List[FileInfo] = []
        for candidate in candidates[: self.MAX_GREP_FILES]:
            try:
                safe_candidate = resolve_context_path(root, candidate)
            except ContextPathOutsideRootError:
                continue
            info = self._file_info(root, safe_candidate)
            if fnmatch.fnmatch(str(info.get("path") or ""), pattern):
                matches.append(info)
        return GlobResult(matches=matches)

    def write(self, file_path: str, content: str) -> WriteResult:
        del content
        return WriteResult(error="ARTIFACT_BACKEND_READ_ONLY", path=file_path)

    def edit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        del old_string, new_string, replace_all
        return EditResult(error="ARTIFACT_BACKEND_READ_ONLY", path=file_path)

    def ls_info(self, path: str) -> List[FileInfo]:
        result = self.ls(path)
        return result.entries or []

    def glob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        result = self.glob(pattern, path)
        return result.matches or []

    def grep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> List[GrepMatch] | str:
        result = self.grep(pattern, path, glob)
        return result.error or result.matches or []

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def agrep(self, pattern: str, path: str | None = None, glob: str | None = None) -> GrepResult:
        return self.grep(pattern, path, glob)

    async def aglob(self, pattern: str, path: str | None = None) -> GlobResult:
        return self.glob(pattern, path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(self, file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def als_info(self, path: str) -> List[FileInfo]:
        return self.ls_info(path)

    async def aglob_info(self, pattern: str, path: str = "/") -> List[FileInfo]:
        return self.glob_info(pattern, path)

    async def agrep_raw(self, pattern: str, path: str | None = None, glob: str | None = None) -> List[GrepMatch] | str:
        return self.grep_raw(pattern, path, glob)


class DeepAgentWorkflowAdapter:
    """API-compatible facade making Deep Agents the outer Diana lead runtime."""

    _ADAPTER_FIELDS = {
        "domain_workflow",
        "lead_llm",
        "semantic_catalog",
        "deep_agent_graph",
        "initialization_error",
        "knowledge_backend",
        "artifact_backend",
        "skill_sources",
        "backend",
        "graph",
    }

    PLAN_GRAPH_TABLE_DETAIL_KIND = "TABLE_DETAIL"
    PLAN_GRAPH_EXACT_EVIDENCE_KINDS = frozenset(
        {
            "METRIC",
            "COLUMN",
            "SCHEMA",
            "TERM",
            "BUSINESS_RULE",
            "RELATIONSHIPS",
            "SEMANTIC_ENTRY",
        }
    )

    MIGRATED_COMPONENTS = (
        "lead_react_loop",
        "run_context_schema",
        "todo_middleware",
        "state_filesystem",
        "domain_artifact_filesystem",
        "artifact_offload",
        "summarization_middleware",
        "patch_tool_calls_middleware",
        "skills_middleware",
        "subagent_middleware",
        "filesystem_permissions",
        "conversation_checkpoint",
    )

    # These domain services have business scoping/lifecycle semantics that a
    # generic editable prompt-memory file cannot preserve. Personal memory may
    # auto-persist through its own gate; shared knowledge is a separate reviewed
    # publish flow. Keep that distinction visible in runtime traces.
    DOMAIN_GOVERNED_COMPONENTS = (
        "merchant_personal_memory_store",
        "shared_knowledge_publish_governance",
        "tenant_authorization_context",
        "topic_workspace_routing",
        "semantic_evidence_acceptance",
        "querygraph_sql_evidence_kernel",
    )

    SYSTEM_PROMPT = """You are Diana's single Core ReAct Agent for merchant analysis.

Operate as one autonomous reason-act-observe loop, with at most the governed action budget returned by the runtime. There is no fixed functional-agent pipeline and Planner is not a subagent.

Runtime contract:
- Call inspect_diana_state first. It returns the current observation and the only governed action ids that are safe now.
- Select one returned action id and call run_diana_action. Use its next observation/catalog to decide again; do not assume a fixed order.
- run_diana_action is the only way to mutate business state, plan/validate QueryGraph, execute tenant-scoped SQL, verify evidence, ask the user, or persist an answer.
- Treat every plan as a revisable hypothesis. After freshness gaps, zero rows, SQL errors, or an unsupported dimension, inspect the observation and independently choose whether to read more knowledge, replan, switch tables, repair/requery, or merge verified results; never run a fixed sequence to completion.
- Zero rows and failed queries are execution observations, never evidence that the business value is zero.
- Never invent metrics, SQL, rows, evidence, or a business answer outside governed tools.

FileSystem-as-Context contract:
- /knowledge is read-only and unavailable until the runtime has narrowed the question to a bounded set of candidate Topics.
- Topic discovery is not a final table decision. The first semantic disclosure is only the candidate Topics' L0 table manifests and business summaries. Compare them, choose the table(s) that cover both the requested fact and dimensions, then read the chosen detailPath.
- Prefer a detail fact table directly when the question asks for a breakdown, ranking, or unsupported dimension. A profile aggregate is optional for cheap/authoritative summaries; detailMetricRef is navigation evidence, never a mandatory profile-then-detail pipeline.
- You are the knowledge-navigation owner. Use Deep Agents' native ls, read_file and grep tools yourself; Planner does not browse or deep-read the knowledge tree.
- Before plan_graph, read the chosen TABLE_DETAIL and at least one exact metric/column/rule/relationship definition or SCHEMA needed by the question. Planner only consumes the trusted refs recorded from your successful read_file calls.
- If plan_graph is rejected for missing semantic evidence, continue with ls/read_file/grep and call plan_graph again; do not ask Planner to search for you.
- Read exact metric/column definitions, schema and business rules only when needed.
- Never infer a formula, physical column, relationship or rule from an index entry alone.
- Scratch files and large tool results are managed by Deep Agents' state filesystem and offload middleware.
- Domain QueryGraph, SQL/result and evidence intermediates are mounted read-only at /artifacts. Use ls/read_file/grep there when the compact observation is insufficient; never assume an artifact's contents without reading it.

Memory and context contract:
- Deep Agents owns short-term conversation messages, checkpoint restore, summarization, the run-scoped context envelope and scratch files.
- Personal merchant memory (preferences, habits and recent focus) remains in the tenant-scoped Diana memory store and may auto-persist through its confidence/privacy/retention gate. Shared metric definitions, rules and terms are separate knowledge candidates and require the knowledge confirm/review/publish flow.
- Tenant scope, Topic routing and semantic evidence remain governed domain state. Consume them only through inspect_diana_state/run_diana_action; never copy either personal memory or shared knowledge into AGENTS.md-style prompt memory.
- Do not write under /memory. Durable personal-memory updates go through Diana's memory lifecycle; shared knowledge updates go through its separate publish lifecycle.

Subagent contract:
- Use task only when a long intermediate process should be isolated or independent work can run in parallel.
- Never delegate routing, planning authority, QueryGraph validation, SQL authority, evidence acceptance or the final answer to a subagent.
- Subagents share the active Topic's read-only /knowledge backend, but cannot call run_diana_action.

When run_diana_action returns terminal=true, stop immediately. The API returns the governed ChatResponse captured by the runtime, never free-form model prose.
"""

    def __init__(self, domain_workflow: Any, lead_llm: Any, semantic_catalog: Any):
        self.domain_workflow = domain_workflow
        self.settings = domain_workflow.settings
        self.lead_llm = lead_llm
        self.semantic_catalog = semantic_catalog
        self.deep_agent_graph: Any = None
        self.initialization_error = ""
        self.knowledge_backend = ReadOnlySemanticBackend(self.semantic_catalog)
        self.artifact_backend = ReadOnlyRunArtifactBackend(self.settings)
        self.skill_sources: List[str] = []
        self.backend = self._build_backend()
        self._inspect_tool = self._build_inspect_tool()
        self._action_tool = self._build_action_tool()
        model = lead_llm.chat_model() if lead_llm and lead_llm.configured else None
        if model is not None:
            try:
                self.deep_agent_graph = create_deep_agent(
                    model=model,
                    tools=[self._inspect_tool, self._action_tool],
                    system_prompt=self.SYSTEM_PROMPT,
                    subagents=[
                        {
                            "name": "general-purpose",
                            "description": (
                                "Isolate a long read-only investigation or parallel evidence synthesis; "
                                "never performs governed planning, SQL, evidence acceptance, or final answering."
                            ),
                            "system_prompt": (
                                "You are an isolated Diana worker. Read only the active Topic knowledge and "
                                "scratch files needed for the assigned subtask. Return a concise finding with "
                                "file/ref evidence. You have no authority to plan QueryGraph, execute SQL, "
                                "accept evidence, or answer the user."
                            ),
                            "tools": [],
                            "skills": self.skill_sources,
                        }
                    ],
                    skills=self.skill_sources or None,
                    permissions=[
                        FilesystemPermission(
                            operations=["write"],
                            paths=["/knowledge", "/knowledge/**", "/skills", "/skills/**"],
                            mode="deny",
                        ),
                        FilesystemPermission(
                            operations=["write"],
                            paths=["/memory", "/memory/**"],
                            mode="deny",
                        ),
                        FilesystemPermission(
                            operations=["write"],
                            paths=["/artifacts", "/artifacts/**"],
                            mode="deny",
                        ),
                    ],
                    backend=self.backend,
                    context_schema=DeepAgentRunContext,
                    checkpointer=self.domain_workflow.checkpoint_manager.saver(),
                    name="diana_lead_agent",
                )
            except Exception as exc:
                self.initialization_error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                LOGGER.exception("Deep Agent initialization failed")
                raise RuntimeError("Deep Agent initialization failed: %s" % self.initialization_error) from exc
        self.graph = self.deep_agent_graph or self.domain_workflow.graph

    def __getattr__(self, name: str) -> Any:
        return getattr(self.domain_workflow, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Preserve the legacy workflow facade for dependency injection.

        ``__getattr__`` already exposes domain services through the adapter, but
        plain assignment would otherwise shadow those services on the facade.
        Tests and runtime composition intentionally replace collaborators such
        as ``memory_store`` and ``planner``; forward those writes to the domain
        workflow while keeping DeepAgent-owned state local.
        """

        domain_workflow = self.__dict__.get("domain_workflow")
        if (
            domain_workflow is None
            or name in self._ADAPTER_FIELDS
            or not hasattr(domain_workflow, name)
        ):
            object.__setattr__(self, name, value)
            return
        setattr(domain_workflow, name, value)

    def _build_backend(self) -> CompositeBackend:
        skill_root = Path(self.settings.resolved_ops_path).parent / "agent_skills"
        routes: Dict[str, Any] = {
            "/knowledge/": self.knowledge_backend,
            "/artifacts/": self.artifact_backend,
        }
        if skill_root.exists():
            routes["/skills/"] = FilesystemBackend(root_dir=skill_root, virtual_mode=True)
            self.skill_sources.append("/skills/")
        return CompositeBackend(default=StateBackend(), routes=routes, artifacts_root="/workspace")

    def _build_inspect_tool(self) -> Any:
        adapter = self

        @tool("inspect_diana_state")
        def inspect_diana_state(runtime: ToolRuntime[DeepAgentRunContext]) -> str:
            """Return the trusted Diana observation and currently allowed action catalog."""

            session = runtime.context.session
            with session.lock:
                payload = adapter._turn_payload(session)
            return json.dumps(payload, ensure_ascii=False, default=str)

        return inspect_diana_state

    def _build_action_tool(self) -> Any:
        adapter = self

        @tool("run_diana_action")
        def run_diana_action(
            action_id: str,
            reason: str,
            runtime: ToolRuntime[DeepAgentRunContext],
        ) -> str:
            """Execute one action id from the latest trusted Diana action catalog."""

            payload = adapter._execute_action(runtime.context.session, action_id, reason)
            return json.dumps(payload, ensure_ascii=False, default=str)

        return run_diana_action

    @staticmethod
    def _active_core_semantic_evidence(session: _DianaLeadSession) -> List[Dict[str, Any]]:
        """Return trusted reads still inside the current Topic workspace."""

        allowed_topics = {
            str(item).strip()
            for item in (session.state.get("topic_workspace") or {}).get("topics") or []
            if str(item).strip()
        }
        with session.lock:
            active = [
                dict(item)
                for item in session.core_semantic_evidence
                if str(item.get("topic") or "") in allowed_topics
                and str(item.get("refId") or "")
                and str(item.get("contentHash") or "")
            ]
            session.core_semantic_evidence = active[-ReadOnlySemanticBackend.MAX_EVIDENCE_ITEMS :]
            return [dict(item) for item in session.core_semantic_evidence]

    def _sync_core_semantic_evidence(self, session: _DianaLeadSession) -> List[Dict[str, Any]]:
        """Synchronize backend-observed Core reads into governed domain state."""

        active = self._active_core_semantic_evidence(session)
        session.state["core_semantic_evidence"] = [dict(item) for item in active]
        session.state["core_managed_filesystem"] = True
        return active

    def _plan_graph_semantic_evidence_gaps(self, session: _DianaLeadSession) -> List[str]:
        """Require L1 table detail plus one compatible exact definition read."""

        evidence = self._active_core_semantic_evidence(session)
        table_details = [
            item
            for item in evidence
            if str(item.get("kind") or "").upper() == self.PLAN_GRAPH_TABLE_DETAIL_KIND
            and str(item.get("contentSnippet") or "").strip()
        ]
        detail_keys = {
            (str(item.get("topic") or ""), str(item.get("table") or ""))
            for item in table_details
            if str(item.get("topic") or "") and str(item.get("table") or "")
        }
        exact_evidence = []
        for item in evidence:
            kind = str(item.get("kind") or "").upper()
            if kind not in self.PLAN_GRAPH_EXACT_EVIDENCE_KINDS:
                continue
            if not str(item.get("contentSnippet") or "").strip():
                continue
            topic = str(item.get("topic") or "")
            table = str(item.get("table") or "")
            if (topic, table) in detail_keys or (
                kind == "RELATIONSHIPS" and any(detail_topic == topic for detail_topic, _ in detail_keys)
            ):
                exact_evidence.append(item)

        gaps: List[str] = []
        if not table_details:
            gaps.append("TABLE_DETAIL_REQUIRED")
        if not exact_evidence:
            gaps.append("EXACT_DEFINITION_OR_SCHEMA_REQUIRED")
        return gaps

    def _prepare_turn(self, session: _DianaLeadSession) -> Dict[str, Any]:
        """Run harness-only gates and expose an unordered safe tool catalog."""

        state = session.state
        state = self.domain_workflow.middleware_chain.after_action(state)
        self.domain_workflow.materialize_plan_clarification(state)
        state = self.domain_workflow.middleware_chain.before_policy(state)
        self.domain_workflow.refresh_execution_tier_policy(state)
        observation = self.domain_workflow.main_agent_observation(state)
        state.setdefault("main_agent_observations", []).append(observation)
        state["main_agent_observations"] = state["main_agent_observations"][-24:]
        state["lead_decision_context"] = self.domain_workflow.build_lead_decision_context(state, observation)
        decision = self.domain_workflow.policy.decide(state)
        available = list(decision.available_actions)
        state["available_actions"] = self.domain_workflow.policy.registry.actions(available)
        session.state = state
        session.observation = observation
        session.available_actions = tuple(available)
        return self._turn_payload(session)

    def _turn_payload(self, session: _DianaLeadSession) -> Dict[str, Any]:
        state = session.state
        catalog = []
        for action_id in session.available_actions:
            action = self.domain_workflow.policy.registry.get(action_id)
            catalog.append(
                {
                    "actionId": action.id,
                    "description": action.description,
                }
            )
        payload: Dict[str, Any] = {
            "status": "TERMINAL" if session.terminal else "ACTION_REQUIRED",
            "terminal": session.terminal,
            "reactRound": int(state.get("react_round") or 0),
            "maxReactRounds": int(self.domain_workflow.policy.max_main_actions),
            "observation": session.observation or {},
            "actionCatalog": catalog,
            "policy": "Choose exactly one actionId from actionCatalog; catalog order is not a workflow.",
        }
        core_evidence = self._active_core_semantic_evidence(session)
        evidence_gaps = self._plan_graph_semantic_evidence_gaps(session)
        payload["coreSemanticEvidence"] = {
            "readCount": len(core_evidence),
            "refs": [
                {
                    key: item.get(key)
                    for key in ["refId", "path", "kind", "topic", "table", "contentHash", "offset"]
                    if item.get(key) not in (None, "")
                }
                for item in core_evidence[-24:]
            ],
            "trustedSource": "successful_core_read_file_calls",
            "planGraphReady": not evidence_gaps,
            "missingForPlanGraph": evidence_gaps,
        }
        topics = [str(item) for item in (state.get("topic_workspace") or {}).get("topics") or [] if str(item)]
        workspace_mode = str((state.get("topic_workspace") or {}).get("mode") or "")
        if topics:
            payload["knowledgeRoots"] = ["/knowledge/topics/%s/manifest.json" % topic for topic in topics]
        artifact_backend = self.__dict__.get("artifact_backend")
        artifact_listing = artifact_backend.ls("/") if artifact_backend is not None else None
        if artifact_listing is not None and not artifact_listing.error and artifact_listing.entries:
            payload["artifactWorkspace"] = {
                "root": "/artifacts",
                "available": True,
                "topLevelEntries": len(artifact_listing.entries),
                "policy": (
                    "Domain intermediates are read-only. Use native ls/read_file/grep to load only the "
                    "QueryGraph, SQL/result or evidence files needed for the next decision."
                ),
            }
        if (
            topics
            and workspace_mode != "clarification_required"
            and not session.table_manifest_disclosed
        ):
            payload["tableManifest"] = build_stable_topic_table_manifest(
                self.semantic_catalog.topic_assets,
                topics,
            )
            payload["semanticDisclosure"] = {
                "layer": "L0",
                "contains": ["topic", "table", "title", "businessSummary", "detailRefId", "detailPath"],
                "omits": ["metrics", "columns", "schema", "rules", "relationships"],
                "next": "choose a table, then read /knowledge/<detailPath>",
            }
            session.table_manifest_disclosed = True
        return payload

    def _execute_action(
        self,
        session: _DianaLeadSession,
        action_id: str,
        reason: str,
        decision_source: str = "deepagent_core_react",
    ) -> Dict[str, Any]:
        with session.lock:
            return self._execute_action_locked(session, action_id, reason, decision_source)

    def _execute_action_locked(
        self,
        session: _DianaLeadSession,
        action_id: str,
        reason: str,
        decision_source: str = "deepagent_core_react",
    ) -> Dict[str, Any]:
        self._sync_core_semantic_evidence(session)
        state = session.state
        selected_id = str(action_id or "").strip()
        if session.terminal:
            return self._turn_payload(session)
        if selected_id not in session.available_actions:
            payload = self._turn_payload(session)
            payload.update(
                {
                    "status": "ACTION_REJECTED",
                    "error": "ACTION_NOT_IN_GOVERNED_CATALOG",
                    "rejectedActionId": selected_id,
                }
            )
            return payload

        if selected_id == "plan_graph":
            evidence_gaps = self._plan_graph_semantic_evidence_gaps(session)
            if evidence_gaps:
                payload = self._turn_payload(session)
                payload.update(
                    {
                        "status": "ACTION_REJECTED",
                        "error": "CORE_SEMANTIC_EVIDENCE_REQUIRED",
                        "rejectedActionId": selected_id,
                        "missingSemanticEvidence": evidence_gaps,
                        "requiredSemanticEvidence": [
                            {
                                "kind": "TABLE_DETAIL",
                                "instruction": "read_file the chosen table detailPath",
                            },
                            {
                                "kind": "EXACT_DEFINITION_OR_SCHEMA",
                                "acceptedKinds": sorted(self.PLAN_GRAPH_EXACT_EVIDENCE_KINDS),
                                "instruction": (
                                    "read_file at least one exact metric/column/rule/relationship definition "
                                    "or schema for that table"
                                ),
                            },
                        ],
                        "next": "Core must continue with native ls/read_file/grep, then call plan_graph again.",
                    }
                )
                return payload

        action = self.domain_workflow.policy.registry.get(selected_id)
        decision = AgentDecision(
            selected_action=action.id,
            selected_node=action.node,
            available_actions=list(session.available_actions),
            reason=str(reason or "DeepAgent Core ReAct selection")[:500],
            observation=str((session.observation or {}).get("summary") or ""),
            source=str(decision_source or "deepagent_core_react"),
        )
        state = self.domain_workflow.middleware_chain.before_action(state, decision)
        if (state.get("terminal_status") or {}).get("active"):
            terminal_action = self.domain_workflow.policy.registry.get("terminal_end")
            decision = AgentDecision(
                selected_action=terminal_action.id,
                selected_node=terminal_action.node,
                available_actions=[terminal_action.id],
                reason="terminal_status became active before action dispatch",
                budget_exhausted=True,
                source="terminal_status",
            )
        state = self.domain_workflow.middleware_chain.capture_action(state, decision)
        state["_next_action"] = decision.selected_node
        state["available_actions"] = self.domain_workflow.policy.registry.actions(decision.available_actions)
        state["agent_decision_reason"] = decision.reason
        self.domain_workflow.ensure_terminal_planning_gap(state, decision)
        state.setdefault("lead_decisions", []).append(decision)
        selected = self.domain_workflow.policy.registry.get(decision.selected_action)
        state.setdefault("action_history", []).append(
            AgentActionTrace(
                round=int(state.get("react_round") or 0),
                action=decision.selected_action,
                node=decision.selected_node,
                agent=selected.agent,
                status="selected",
                reason=decision.reason,
                available_actions=decision.available_actions,
                observation=decision.observation,
            )
        )
        emit(
            state,
            "agent.action.selected",
            "RUNTIME_FAIL_CLOSED" if decision.source == "runtime_fail_closed" else "DEEP_AGENT_CORE_REACT",
            {
                "action": decision.selected_action,
                "node": decision.selected_node,
                "reactRound": int(state.get("react_round") or 0),
                "availableActions": decision.available_actions,
                "reason": decision.reason,
                "source": decision.source,
            },
        )
        handler = getattr(self.domain_workflow, decision.selected_node)
        session.state = handler(state)
        session.action_count += 1

        terminal_ids = {"ask_human", "cache_answer", "terminal_end"}
        if decision.selected_action in terminal_ids:
            session.state = self.domain_workflow.finalize_action_contract(session.state)
            session.terminal = True
            session.available_actions = ()
            session.observation = self.domain_workflow.main_agent_observation(session.state)
            session.sink.response = self.domain_workflow.to_response(session.state)
            self.domain_workflow.schedule_post_answer_tail(session.state)
            payload = self._turn_payload(session)
            payload["responseId"] = session.sink.response.id
            payload["persisted"] = session.sink.response.persisted
            return payload
        return self._prepare_turn(session)

    def _bootstrap_session(
        self,
        question: str,
        merchant_id: str,
        context: Optional[ChatContext],
        listener: Any,
        thread_id: str,
        run_id: str,
        message_history: Optional[List[ConversationMessage]],
        sink: _ResultSink,
    ) -> _DianaLeadSession:
        state = self.domain_workflow._initial_state(
            question,
            merchant_id,
            context,
            listener,
            thread_id,
            run_id,
            message_history,
        )
        state = self.domain_workflow.preflight_route(state)
        if self.domain_workflow.preflight_needs_full_context(state):
            state = self.domain_workflow.inherit_context(state)
            state = self.domain_workflow.runtime_bootstrap(state)
        session = _DianaLeadSession(state=state, sink=sink)
        self._prepare_turn(session)
        return session

    def _finish_without_model(self, session: _DianaLeadSession) -> None:
        """Use the existing fail-closed policy if the Core model stops early."""

        safety_budget = int(self.domain_workflow.policy.max_main_actions) + 4
        while not session.terminal and session.action_count < safety_budget:
            decision = self.domain_workflow.policy.decide(session.state)
            available = list(decision.available_actions)
            if not available:
                break
            selected = decision.selected_action
            if selected == "lead_arbitrate":
                selected, _, _ = self.domain_workflow.runtime_safe_lead_recovery_action(
                    session.state,
                    decision,
                    available,
                )
            if selected not in session.available_actions:
                selected = available[0]
            self._execute_action(
                session,
                selected,
                "runtime fail-closed continuation after DeepAgent stopped",
                decision_source="runtime_fail_closed",
            )
        if session.terminal:
            return
        state = session.state
        state["human_clarification_required"] = True
        state["human_clarification_question"] = "主 Agent 未能安全完成本轮分析，请补充更明确的分析目标后重试。"
        state["human_clarification_stage"] = "LEAD_DECISION"
        state["human_clarification_type"] = "lead_decision_unavailable"
        session.available_actions = ("ask_human",)
        self._execute_action(
            session,
            "ask_human",
            "fail closed after DeepAgent termination",
            decision_source="runtime_fail_closed",
        )

    def run(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Any = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[List[ConversationMessage]] = None,
    ) -> ChatResponse:
        if self.deep_agent_graph is None:
            return self.domain_workflow.run(
                question,
                merchant_id,
                context,
                listener,
                thread_id,
                run_id,
                message_history,
            )
        actual_thread_id = thread_id or "thread_" + uuid.uuid4().hex
        actual_run_id = run_id or "run_" + uuid.uuid4().hex
        actual_merchant_id = merchant_id or self.settings.merchant_id
        sink = _ResultSink()
        session: Optional[_DianaLeadSession] = None
        register_event_listener(actual_run_id, listener)
        try:
            with tool_runtime_scope(actual_merchant_id, actual_thread_id, actual_run_id):
                session = self._bootstrap_session(
                    question=(question or "").strip(),
                    merchant_id=actual_merchant_id,
                    context=context,
                    listener=listener,
                    thread_id=actual_thread_id,
                    run_id=actual_run_id,
                    message_history=message_history,
                    sink=sink,
                )
                request = DeepAgentRunContext(
                    question=(question or "").strip(),
                    merchant_id=actual_merchant_id,
                    chat_context=context,
                    listener=listener,
                    thread_id=actual_thread_id,
                    run_id=actual_run_id,
                    message_history=tuple(message_history or ()),
                    sink=sink,
                    session=session,
                )
                with self.knowledge_backend.scope_to_session(session):
                    try:
                        self.deep_agent_graph.invoke(
                            {"messages": [{"role": "user", "content": request.question}]},
                            config=self.domain_workflow.checkpoint_manager.config_for_deep_agent(actual_thread_id, actual_run_id),
                            context=request,
                        )
                    except Exception as exc:
                        if sink.response is None:
                            LOGGER.warning("Deep Agent Core loop stopped before governed completion: %s", str(exc)[:300])
                    if sink.response is None:
                        self._finish_without_model(session)
        except Exception as exc:
            LOGGER.exception("Diana governed runtime failed: %s", str(exc)[:300])
            raise
        finally:
            unregister_event_listener(actual_run_id)
        if sink.response is None:
            raise RuntimeError("Diana Core Agent ended without a governed ChatResponse")
        return self._attach_deep_agent_trace(sink.response, actual_thread_id, actual_run_id)

    async def run_async(
        self,
        question: str,
        merchant_id: str = "",
        context: Optional[ChatContext] = None,
        listener: Any = None,
        thread_id: str = "",
        run_id: str = "",
        message_history: Optional[List[ConversationMessage]] = None,
    ) -> ChatResponse:
        return await asyncio.to_thread(
            self.run,
            question,
            merchant_id,
            context,
            listener,
            thread_id,
            run_id,
            message_history,
        )

    def _attach_deep_agent_trace(self, response: ChatResponse, thread_id: str, run_id: str) -> ChatResponse:
        trace = dict(response.debug_trace or {})
        harness = dict(trace.get("harness") or {})
        domain_checkpoint = harness.get("checkpoint")
        if domain_checkpoint:
            harness["domainCheckpoint"] = domain_checkpoint
        harness["checkpoint"] = self.domain_workflow.checkpoint_manager.deep_agent_ref(thread_id, run_id)
        harness["runtime"] = "deepagent"
        harness["migratedComponents"] = list(self.MIGRATED_COMPONENTS)
        harness["domainGovernedComponents"] = list(self.DOMAIN_GOVERNED_COMPONENTS)
        trace["harness"] = harness
        return response.model_copy(update={"debug_trace": trace})
