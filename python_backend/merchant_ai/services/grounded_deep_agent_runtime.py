from __future__ import annotations

import fnmatch
import hashlib
import json
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage

try:
    from deepagents import FilesystemPermission, create_deep_agent
    from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
    from deepagents.backends.protocol import (
        EditResult,
        FileInfo,
        GlobResult,
        GrepMatch,
        GrepResult,
        LsResult,
        ReadResult,
        WriteResult,
    )

    _DEEPAGENTS_IMPORT_ERROR = ""
except ImportError as exc:  # pragma: no cover - exercised through constructor gate
    create_deep_agent = None
    _DEEPAGENTS_IMPORT_ERROR = str(exc)

    @dataclass
    class FilesystemPermission:  # type: ignore[no-redef]
        operations: list[str]
        paths: list[str]
        mode: str

    @dataclass
    class FileInfo:  # type: ignore[no-redef]
        path: str
        is_dir: bool = False
        size: int = 0
        modified_at: str = ""

    @dataclass
    class LsResult:  # type: ignore[no-redef]
        entries: list[FileInfo] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class ReadResult:  # type: ignore[no-redef]
        file_data: Optional[dict[str, Any]] = None
        error: Optional[str] = None

    @dataclass
    class GrepMatch:  # type: ignore[no-redef]
        path: str
        line: int
        text: str

    @dataclass
    class GrepResult:  # type: ignore[no-redef]
        matches: list[GrepMatch] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class GlobResult:  # type: ignore[no-redef]
        matches: list[Any] = field(default_factory=list)
        error: Optional[str] = None

    @dataclass
    class WriteResult:  # type: ignore[no-redef]
        error: Optional[str] = None
        path: str = ""

    @dataclass
    class EditResult:  # type: ignore[no-redef]
        error: Optional[str] = None
        path: str = ""

    class StateBackend:  # type: ignore[no-redef]
        pass

    class FilesystemBackend:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs

    class CompositeBackend:  # type: ignore[no-redef]
        def __init__(self, **kwargs: Any):
            self.kwargs = kwargs
try:
    from langchain.tools import ToolRuntime, tool
except ImportError:  # pragma: no cover - compatibility for minimal test runtime
    from langchain_core.tools import tool

    _ContextT = TypeVar("_ContextT")

    class ToolRuntime(Generic[_ContextT]):  # type: ignore[no-redef]
        context: _ContextT

from merchant_ai.models import ChatResponse, MerchantInfo, RecallBundle
from merchant_ai.services.assets import normalize_semantic_path
from merchant_ai.services.grounded_runtime_kernel import (
    GroundedRuntimeKernel,
    GroundedRuntimeSession,
)
from merchant_ai.services.grounded_query_contract import GroundedBindingHints


_SEMANTIC_SCOPE: ContextVar[Optional["GroundedDeepAgentSession"]] = ContextVar(
    "grounded_deep_agent_semantic_scope",
    default=None,
)


@dataclass
class GroundedDeepAgentSession:
    runtime: GroundedRuntimeSession
    core_semantic_evidence: list[dict[str, Any]] = field(default_factory=list)
    opened_topics: list[str] = field(default_factory=list)
    topic_index_read: bool = False
    lock: Any = field(default_factory=RLock, repr=False)

    def effective_topics(self) -> list[str]:
        return list(
            dict.fromkeys(
                [*self.runtime.workspace_topics, *self.opened_topics]
            )
        )


@dataclass(frozen=True)
class GroundedDeepAgentRunContext:
    thread_id: str
    run_id: str
    session: GroundedDeepAgentSession


class GroundedSemanticBackend:
    """Read-only semantic filesystem scoped to one Grounded Core session."""

    MAX_EVIDENCE_ITEMS = 64

    def __init__(
        self,
        semantic_catalog: Any,
        *,
        reader_is_core: Optional[Callable[[], bool]] = None,
    ):
        self.semantic_catalog = semantic_catalog
        self.reader_is_core = reader_is_core or _deepagent_reader_is_core

    @contextmanager
    def scope(self, session: GroundedDeepAgentSession) -> Iterator[None]:
        token = _SEMANTIC_SCOPE.set(session)
        try:
            yield
        finally:
            _SEMANTIC_SCOPE.reset(token)

    @staticmethod
    def _session() -> Optional[GroundedDeepAgentSession]:
        return _SEMANTIC_SCOPE.get()

    @staticmethod
    def _path(value: str) -> str:
        return normalize_semantic_path(str(value or "/")).strip("/")

    @staticmethod
    def _topic(path: str) -> str:
        parts = [item for item in str(path or "").split("/") if item]
        return parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""

    @staticmethod
    def _manifest_topic(path: str) -> str:
        parts = [item for item in str(path or "").split("/") if item]
        if len(parts) == 3 and parts[0] == "topics" and parts[2] == "manifest.json":
            return parts[1]
        return ""

    def _scope_error(self, path: str) -> str:
        session = self._session()
        if session is None:
            return "GROUNDED_SESSION_REQUIRED"
        normalized = self._path(path)
        if normalized.endswith("/asset.json"):
            return (
                "FULL_TABLE_ASSET_DENIED: read the table detail.json, then only "
                "the required metrics/columns/schema/relationships children"
            )
        if normalized in {"", "topics", "topics/index.json"}:
            return ""
        manifest_topic = self._manifest_topic(normalized)
        if manifest_topic:
            if manifest_topic in session.effective_topics() or session.topic_index_read:
                return ""
            return "TOPIC_INDEX_READ_REQUIRED"
        topic = self._topic(normalized)
        if topic and topic not in session.effective_topics():
            return "TOPIC_SCOPE_DENIED:%s" % topic
        return ""

    def ls(self, path: str) -> LsResult:
        normalized = self._path(path)
        scope_error = self._scope_error(normalized)
        if scope_error:
            return LsResult(error=scope_error)
        session = self._session()
        if normalized in {"", "topics"}:
            topics = session.effective_topics() if session else []
            return LsResult(
                entries=[
                    FileInfo(path="/topics/index.json", is_dir=False, size=0, modified_at=""),
                    *[
                        FileInfo(
                            path="/topics/%s/" % topic_name,
                            is_dir=True,
                            size=0,
                            modified_at="",
                        )
                        for topic_name in topics
                    ],
                ]
            )
        try:
            items = self.semantic_catalog.ls(path=normalized, limit=500)
        except Exception as exc:
            return LsResult(error="SEMANTIC_LS_FAILED:%s" % str(exc)[:300])
        return LsResult(
            entries=[
                FileInfo(
                    path="/" + str(item.get("path") or "").lstrip("/"),
                    is_dir=False,
                    size=int(item.get("estimatedChars") or 0),
                    modified_at="",
                )
                for item in items
                if isinstance(item, dict)
                and item.get("path")
                and not str(item.get("path") or "").endswith("/asset.json")
            ]
        )

    def read(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        normalized = self._path(file_path)
        scope_error = self._scope_error(normalized)
        if scope_error:
            return ReadResult(error=scope_error)
        try:
            result = self.semantic_catalog.read(
                path=normalized,
                max_chars=2_000_000,
                offset=0,
            )
        except Exception as exc:
            return ReadResult(error="SEMANTIC_READ_FAILED:%s" % str(exc)[:300])
        if not isinstance(result, dict) or not result.get("success"):
            return ReadResult(error=str((result or {}).get("error") or "SEMANTIC_REF_NOT_FOUND"))

        session = self._session()
        kind = str(result.get("kind") or "").upper()
        topic_name = str(result.get("topic") or "")
        if session is not None:
            if kind == "TOPIC_INDEX":
                session.topic_index_read = True
            elif kind == "TOPIC_MANIFEST" and topic_name not in session.effective_topics():
                if not session.topic_index_read:
                    return ReadResult(error="TOPIC_INDEX_READ_REQUIRED")
                session.opened_topics.append(topic_name)
                if topic_name not in session.runtime.workspace_topics:
                    session.runtime.workspace_topics.append(topic_name)

        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        content = "".join(lines[start:end])
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if (
            session is not None
            and complete
            and kind not in {"TOPIC_INDEX", "TOPIC_MANIFEST"}
            and self.reader_is_core()
        ):
            evidence = {
                "refId": str(result.get("refId") or ""),
                "path": str(result.get("path") or normalized).lstrip("/"),
                "kind": kind,
                "topic": topic_name,
                "table": str(result.get("table") or ""),
                "contentSnippet": full_content,
                "contentHash": hashlib.sha256(full_content.encode("utf-8")).hexdigest(),
                "contentComplete": True,
                "offset": 0,
            }
            if evidence["refId"] and topic_name in session.effective_topics():
                with session.lock:
                    retained = [
                        item
                        for item in session.core_semantic_evidence
                        if item.get("refId") != evidence["refId"]
                    ]
                    retained.append(evidence)
                    session.core_semantic_evidence = retained[-self.MAX_EVIDENCE_ITEMS :]
        return ReadResult(file_data={"content": content, "encoding": "utf-8"})

    def record_core_read(
        self,
        session: GroundedDeepAgentSession,
        file_path: str,
        *,
        offset: int = 0,
        limit: int = 2000,
    ) -> bool:
        """Record one successful root-Core read without relying on thread config.

        Deep Agents may execute filesystem backends in a worker context where
        ``langgraph.config.get_config()`` is unavailable.  Root/subagent
        authority is therefore decided by main-agent tool middleware; this
        method only verifies that the exact semantic file was read completely.
        """

        normalized = self._path(file_path)
        if normalized == "knowledge":
            normalized = ""
        elif normalized.startswith("knowledge/"):
            normalized = normalized[len("knowledge/") :]
        try:
            result = self.semantic_catalog.read(
                path=normalized,
                max_chars=2_000_000,
                offset=0,
            )
        except Exception:
            return False
        if not isinstance(result, dict) or not result.get("success"):
            return False
        kind = str(result.get("kind") or "").upper()
        topic_name = str(result.get("topic") or "")
        if kind in {"TOPIC_INDEX", "TOPIC_MANIFEST"}:
            return False
        full_content = str(result.get("content") or "")
        lines = full_content.splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        if not complete or topic_name not in session.effective_topics():
            return False
        evidence = {
            "refId": str(result.get("refId") or ""),
            "path": str(result.get("path") or normalized).lstrip("/"),
            "kind": kind,
            "topic": topic_name,
            "table": str(result.get("table") or ""),
            "contentSnippet": full_content,
            "contentHash": hashlib.sha256(full_content.encode("utf-8")).hexdigest(),
            "contentComplete": True,
            "offset": 0,
        }
        if not evidence["refId"]:
            return False
        with session.lock:
            retained = [
                item
                for item in session.core_semantic_evidence
                if item.get("refId") != evidence["refId"]
            ]
            retained.append(evidence)
            session.core_semantic_evidence = retained[-self.MAX_EVIDENCE_ITEMS :]
        return True

    def grep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        normalized = self._path(path or "")
        scope_error = self._scope_error(normalized)
        if scope_error:
            return GrepResult(error=scope_error)
        session = self._session()
        topics = [self._topic(normalized)] if self._topic(normalized) else (
            session.effective_topics() if session else []
        )
        hits: list[dict[str, Any]] = []
        try:
            for topic_name in topics:
                hits.extend(
                    self.semantic_catalog.grep(
                        query=pattern,
                        topic=topic_name,
                        limit=100,
                        path=normalized if normalized not in {"", "topics"} else "",
                    )
                )
        except Exception as exc:
            return GrepResult(error="SEMANTIC_GREP_FAILED:%s" % str(exc)[:300])
        matches: list[GrepMatch] = []
        for hit in hits:
            hit_path = "/" + str(hit.get("path") or "").lstrip("/")
            if hit_path.endswith("/asset.json"):
                continue
            if glob and not fnmatch.fnmatch(hit_path, glob):
                continue
            snippets = hit.get("snippets") or [hit.get("summary") or hit.get("title") or ""]
            matches.extend(
                GrepMatch(path=hit_path, line=1, text=str(snippet)[:1000])
                for snippet in snippets[:3]
            )
        return GrepResult(matches=matches[:100])

    def glob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        listing = self.ls(path or "/")
        if listing.error:
            return GlobResult(error=listing.error)
        return GlobResult(
            matches=[
                item
                for item in listing.entries or []
                if fnmatch.fnmatch(
                    str(item.get("path") if isinstance(item, dict) else item.path),
                    pattern,
                )
            ]
        )

    @staticmethod
    def write(file_path: str, content: str) -> WriteResult:
        return WriteResult(error="READ_ONLY_SEMANTIC_FILESYSTEM", path=file_path)

    @staticmethod
    def edit(
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return EditResult(error="READ_ONLY_SEMANTIC_FILESYSTEM", path=file_path)

    def ls_info(self, path: str) -> list[FileInfo]:
        result = self.ls(path)
        return [] if result.error else list(result.entries)

    def glob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        result = self.glob(pattern, path)
        return [] if result.error else list(result.matches or [])

    def grep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> Any:
        result = self.grep(pattern, path, glob)
        return result.error or list(result.matches)

    async def als(self, path: str) -> LsResult:
        return self.ls(path)

    async def aread(self, file_path: str, offset: int = 0, limit: int = 2000) -> ReadResult:
        return self.read(file_path, offset, limit)

    async def agrep(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> GrepResult:
        return self.grep(pattern, path, glob)

    async def aglob(self, pattern: str, path: Optional[str] = None) -> GlobResult:
        return self.glob(pattern, path)

    async def awrite(self, file_path: str, content: str) -> WriteResult:
        return self.write(file_path, content)

    async def aedit(
        self,
        file_path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> EditResult:
        return self.edit(file_path, old_string, new_string, replace_all)

    async def als_info(self, path: str) -> list[FileInfo]:
        return self.ls_info(path)

    async def aglob_info(self, pattern: str, path: str = "/") -> list[FileInfo]:
        return self.glob_info(pattern, path)

    async def agrep_raw(
        self,
        pattern: str,
        path: Optional[str] = None,
        glob: Optional[str] = None,
    ) -> Any:
        return self.grep_raw(pattern, path, glob)


class GroundedCoreToolBoundaryMiddleware(AgentMiddleware):
    """Bind semantic-read authority to the root Core's actual tool calls."""

    name = "GroundedCoreToolBoundaryMiddleware"

    def __init__(self, semantic_backend: GroundedSemanticBackend):
        self.semantic_backend = semantic_backend

    def wrap_tool_call(self, request: Any, handler: Callable[[Any], Any]) -> Any:
        tool_call = dict(getattr(request, "tool_call", None) or {})
        tool_name = str(tool_call.get("name") or "")
        tool_call_id = str(tool_call.get("id") or "")
        if tool_name == "task":
            return ToolMessage(
                content=(
                    "SubAgent dispatch is disabled in this grounded runtime until "
                    "worker evidence acceptance is independently auditable."
                ),
                name=tool_name,
                tool_call_id=tool_call_id,
                status="error",
            )

        result = handler(request)
        if tool_name != "read_file" or getattr(result, "status", "success") == "error":
            return result
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        session = getattr(context, "session", None)
        if not isinstance(session, GroundedDeepAgentSession):
            return result
        args = dict(tool_call.get("args") or {})
        self.semantic_backend.record_core_read(
            session,
            str(args.get("file_path") or ""),
            offset=int(args.get("offset") or 0),
            limit=int(args.get("limit") or 2000),
        )
        return result


class GroundedDeepAgentRuntime:
    """Single DeepAgent Core backed only by GroundedRuntimeKernel tools."""

    SYSTEM_PROMPT = """You are the single Grounded merchant-analysis Core.

The first user message already contains the automatically selected Topic L0 manifest and one Topic-scoped thin recall. Recall is navigation evidence, never planning authority.
The first user message also contains trustedExecutionScope. It is authoritative runtime state, not a user claim. When merchantScopeBound=true, never ask the user for merchant_id and never propose bypassing tenant filtering; the executor binds the declared merchant scope automatically.
Use native ls/read_file/grep progressively under /knowledge. Read exact table detail, metric, column and relationship files before proposing bindings.
Published metric files already contain the governed formula, source columns, unit and time semantics. When metricRefs satisfy the question, do not also submit fieldAggregations for the same measures.
For a simple same-table scalar metric query, the expected disclosure path is table detail plus the exact metric files. Do not read schema, columns/index.json, the time column, or metric source-column files unless the question needs a field aggregation, business dimension, filter, join, or a published metric is unavailable.
GroundedQueryContract is the only planning authority. A candidate may be revised; only a READY candidate that compiles successfully becomes active.
propose_grounded_contract.binding_hints has a strict schema. Use only tableRefs, metricRefs, fieldAggregations, dimensionRefs, groupByRef, labelRefs, relationshipRefs, ranking, analysisMode and timeExpression; never invent alternative keys such as tableRef, metricBindings, metrics, timeWindow or timeRange.
Available governed tools are retrieve_knowledge, propose_grounded_contract, execute_grounded_query, compose_verified_answer and ask_human. There is no action catalog and no legacy planner.
Use retrieve_knowledge only for a targeted supplemental query; it remains inside the active Topic workspace. Use the filesystem to open another Topic only after reading /knowledge/topics/index.json.
Do not call task in this runtime. SubAgent dispatch is disabled until worker evidence acceptance is independently auditable.
Never invent a formula, binding, SQL result, evidence status or answer. Finish only after compose_verified_answer or ask_human succeeds.
"""

    def __init__(
        self,
        kernel: GroundedRuntimeKernel,
        lead_model: Any,
        semantic_catalog: Any,
        *,
        checkpointer: Any = None,
        checkpoint_config_factory: Optional[
            Callable[[str, str], dict[str, Any]]
        ] = None,
        skill_root: Optional[str] = None,
        agent_factory: Any = None,
        backend: Any = None,
    ):
        self.kernel = kernel
        self.semantic_catalog = semantic_catalog
        self.checkpointer = checkpointer
        self.checkpoint_config_factory = checkpoint_config_factory
        # Native backend reads provide content only. Root-Core evidence authority
        # is recorded by tool middleware, never by ambient thread-local config.
        self.knowledge_backend = GroundedSemanticBackend(
            semantic_catalog,
            reader_is_core=lambda: False,
        )
        self.core_tool_boundary = GroundedCoreToolBoundaryMiddleware(
            self.knowledge_backend
        )
        self.skill_sources: list[str] = []
        self.backend = backend or self._build_backend(skill_root)
        self.tools = self._build_tools()
        self.initialization_error = ""
        self.deep_agent_graph: Any = None
        model = self._resolve_model(lead_model)
        if model is None:
            raise RuntimeError("Grounded DeepAgent initialization failed: model is not configured")
        if agent_factory is None:
            if create_deep_agent is None:
                raise RuntimeError(
                    "Grounded DeepAgent initialization failed: deepagents unavailable: %s"
                    % _DEEPAGENTS_IMPORT_ERROR
                )
            agent_factory = create_deep_agent
        try:
            self.deep_agent_graph = agent_factory(
                model=model,
                tools=self.tools,
                system_prompt=self.SYSTEM_PROMPT,
                middleware=[self.core_tool_boundary],
                subagents=[
                    {
                        # deepagents 0.6.x otherwise appends a default worker
                        # named general-purpose which inherits every custom
                        # tool.  Override that exact identity with an explicit
                        # zero-custom-tool read-only worker.
                        "name": "general-purpose",
                        "description": "Read-only isolated semantic investigation for parallel evidence gathering.",
                        "system_prompt": (
                            "Use only native read-only filesystem tools. Return refs and concise findings. "
                            "Do not route, propose a Contract, execute SQL, verify evidence, answer, or ask the user."
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
                    )
                ],
                backend=self.backend,
                context_schema=GroundedDeepAgentRunContext,
                checkpointer=checkpointer,
                name="grounded_merchant_core",
            )
        except Exception as exc:
            self.initialization_error = "%s:%s" % (type(exc).__name__, str(exc)[:500])
            raise RuntimeError(
                "Grounded DeepAgent initialization failed: %s" % self.initialization_error
            ) from exc
        if self.deep_agent_graph is None:
            raise RuntimeError("Grounded DeepAgent initialization failed: agent factory returned no graph")

    @staticmethod
    def _resolve_model(value: Any) -> Any:
        if value is None:
            return None
        if hasattr(value, "configured") and not bool(value.configured):
            return None
        if hasattr(value, "chat_model"):
            return value.chat_model()
        return value

    def _build_backend(self, skill_root: Optional[str]) -> CompositeBackend:
        routes: dict[str, Any] = {"/knowledge/": self.knowledge_backend}
        if skill_root and Path(skill_root).exists():
            routes["/skills/"] = FilesystemBackend(
                root_dir=Path(skill_root),
                virtual_mode=True,
            )
            self.skill_sources.append("/skills/")
        return CompositeBackend(
            default=StateBackend(),
            routes=routes,
            artifacts_root="/workspace",
        )

    def _build_tools(self) -> list[Any]:
        runtime_owner = self

        @tool("retrieve_knowledge")
        def retrieve_knowledge(
            query: str,
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Run one targeted supplemental recall inside the active Topic scope."""

            bundle = runtime_owner.kernel.recall_navigation(
                runtime.context.session.runtime,
                query=str(query or "").strip(),
            )
            return json.dumps(
                {
                    "status": "OK",
                    "reason": str(reason or "")[:500],
                    "recallCandidates": _thin_recall(bundle, limit=8),
                    "scope": runtime.context.session.effective_topics(),
                },
                ensure_ascii=False,
            )

        @tool("propose_grounded_contract")
        def propose_grounded_contract(
            read_ref_ids: list[str],
            binding_hints: GroundedBindingHints,
            auto_compile: bool,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Propose from exact Core reads using the strict typed BindingHints schema."""

            session = runtime.context.session
            if not isinstance(binding_hints, GroundedBindingHints):
                binding_hints = GroundedBindingHints.model_validate(binding_hints)
            requested = list(
                dict.fromkeys(
                    _canonical_progressive_ref(str(item or "").strip())
                    for item in read_ref_ids
                    if str(item or "").strip()
                )
            )
            binding_hints = binding_hints.model_copy(
                update={
                    "table_refs": [
                        _canonical_progressive_ref(ref_id)
                        for ref_id in binding_hints.table_refs
                    ]
                }
            )
            with session.lock:
                evidence_by_ref = {
                    str(item.get("refId") or ""): dict(item)
                    for item in session.core_semantic_evidence
                    if str(item.get("refId") or "")
                }
            missing = [ref_id for ref_id in requested if ref_id not in evidence_by_ref]
            if missing:
                read_next: list[dict[str, str]] = []
                for ref_id in missing:
                    try:
                        resolved = runtime_owner.semantic_catalog.read(
                            ref_id=ref_id,
                            max_chars=1,
                            offset=0,
                        )
                    except Exception:
                        resolved = {}
                    read_next.append(
                        {
                            "refId": ref_id,
                            "path": str((resolved or {}).get("path") or ""),
                        }
                    )
                return json.dumps(
                    {
                        "status": "BLOCKED",
                        "code": "SEMANTIC_REF_NOT_READ",
                        "missingRefs": missing,
                        "readNext": read_next,
                        "instruction": (
                            "Read each non-empty /knowledge path completely, then resubmit "
                            "the same typed binding_hints."
                        ),
                    },
                    ensure_ascii=False,
                )
            evidence = [evidence_by_ref[ref_id] for ref_id in requested]
            attempt = runtime_owner.kernel.propose_contract(
                session.runtime,
                evidence,
                binding_hints,
                topics=session.effective_topics(),
            )
            if attempt.contract.ready and auto_compile:
                attempt = runtime_owner.kernel.compile_candidate(
                    session.runtime,
                    attempt.attempt_id,
                )
            return json.dumps(
                {
                    "attemptId": attempt.attempt_id,
                    "status": attempt.contract.status,
                    "queryShape": attempt.contract.query_shape,
                    "compileStatus": attempt.compile_status,
                    "activated": attempt.activated,
                    "acceptedBindingHints": attempt.contract.binding_hints.model_dump(
                        by_alias=True
                    ),
                    "gaps": [
                        gap.model_dump(by_alias=True)
                        for gap in attempt.contract.unresolved_gaps
                    ],
                    "rejectedBindings": [
                        item.model_dump(by_alias=True)
                        for item in attempt.contract.rejected_bindings
                    ],
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("execute_grounded_query")
        def execute_grounded_query(
            reason: str,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Execute the active compiled Contract and immediately verify evidence."""

            session = runtime.context.session.runtime
            run_result = runtime_owner.kernel.execute_active(
                session,
                run_id=runtime.context.run_id,
            )
            verified = runtime_owner.kernel.verify_active(session)
            return json.dumps(
                {
                    "status": "VERIFIED" if verified.passed else "VERIFICATION_GAPPED",
                    "reason": str(reason or "")[:500],
                    "rowCount": len(run_result.merged_query_bundle.rows),
                    "tables": list(run_result.merged_query_bundle.tables),
                    "blockingGaps": [
                        gap.model_dump(by_alias=True) for gap in verified.blocking_gaps
                    ],
                    "warningGaps": [
                        gap.model_dump(by_alias=True) for gap in verified.warning_gaps
                    ],
                },
                ensure_ascii=False,
                default=str,
            )

        @tool("compose_verified_answer")
        def compose_verified_answer(
            allow_llm: bool,
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Compose and store the final answer from verified evidence only."""

            answer = runtime_owner.kernel.compose_answer(
                runtime.context.session.runtime,
                allow_llm=allow_llm,
            )
            return json.dumps(
                {"status": "ANSWERED", "answer": answer},
                ensure_ascii=False,
            )

        @tool("ask_human")
        def ask_human(
            question: str,
            stage: str,
            clarification_type: str,
            options: list[str],
            runtime: ToolRuntime[GroundedDeepAgentRunContext],
        ) -> str:
            """Create a typed human clarification and stop query execution."""

            normalized_type = str(clarification_type or "").strip().upper()
            if normalized_type.startswith(("SYSTEM_", "INTERNAL_", "TOOL_", "COMPILER_")):
                return json.dumps(
                    {
                        "status": "REJECTED",
                        "code": "INTERNAL_FAILURE_IS_NOT_USER_CLARIFICATION",
                        "message": (
                            "ask_human is reserved for genuinely missing user business input; "
                            "use the trusted runtime scope and return an operational failure for internal blockers"
                        ),
                    },
                    ensure_ascii=False,
                )

            request = runtime_owner.kernel.request_clarification(
                runtime.context.session.runtime,
                question,
                stage=stage,
                clarification_type=clarification_type,
                options=options,
            )
            return json.dumps(
                {"status": "CLARIFICATION_REQUIRED", "clarification": request.model_dump(by_alias=True)},
                ensure_ascii=False,
            )

        return [
            retrieve_knowledge,
            propose_grounded_contract,
            execute_grounded_query,
            compose_verified_answer,
            ask_human,
        ]

    def run(
        self,
        question: str,
        merchant_id: str,
        *,
        merchant: Optional[MerchantInfo] = None,
        access_role: str = "merchant_analyst",
        user_scope: Optional[dict[str, Any]] = None,
        thread_id: str = "",
        run_id: str = "",
    ) -> ChatResponse:
        actual_thread_id = thread_id or "thread_%s" % uuid.uuid4().hex
        actual_run_id = run_id or "run_%s" % uuid.uuid4().hex
        kernel_session = self.kernel.new_session(
            question,
            merchant_id,
            merchant=merchant,
            access_role=access_role,
            user_scope=user_scope,
        )
        self.kernel.route_topic(kernel_session)
        self.kernel.recall_navigation(kernel_session)
        session = GroundedDeepAgentSession(runtime=kernel_session)
        context = GroundedDeepAgentRunContext(
            thread_id=actual_thread_id,
            run_id=actual_run_id,
            session=session,
        )
        first_context = self._initial_context(session)
        if self.checkpoint_config_factory is not None:
            config = self.checkpoint_config_factory(actual_thread_id, actual_run_id)
        elif self.checkpointer is not None:
            config = {
                "configurable": {
                    "thread_id": actual_thread_id,
                    "run_id": actual_run_id,
                }
            }
        else:
            config = None
        with self.knowledge_backend.scope(session):
            self.deep_agent_graph.invoke(
                {"messages": [{"role": "user", "content": first_context}]},
                config=config,
                context=context,
            )
        return self._governed_response(session, actual_thread_id, actual_run_id)

    def _initial_context(self, session: GroundedDeepAgentSession) -> str:
        manifests: list[dict[str, Any]] = []
        for topic_name in session.runtime.workspace_topics:
            result = self.semantic_catalog.read(
                path="topics/%s/manifest.json" % topic_name,
                max_chars=80_000,
                offset=0,
            )
            if not isinstance(result, dict) or not result.get("success"):
                raise RuntimeError(
                    "Grounded bootstrap failed: Topic L0 manifest unavailable for %s"
                    % topic_name
                )
            manifests.append(
                {
                    "topic": topic_name,
                    "refId": str(result.get("refId") or ""),
                    "path": str(result.get("path") or ""),
                    "content": str(result.get("content") or ""),
                }
            )
        if not manifests:
            raise RuntimeError("Grounded bootstrap failed: no routed Topic L0 manifest")
        payload = {
            "question": session.runtime.question,
            "trustedExecutionScope": {
                "merchantScopeBound": bool(session.runtime.merchant_id),
                "merchantId": session.runtime.merchant_id,
                "merchantName": str(session.runtime.merchant.merchant_name or ""),
                "accessRole": session.runtime.access_role,
                "authorizedStoreIds": [
                    str(item)
                    for item in (
                        session.runtime.user_scope.get("storeIds")
                        or session.runtime.user_scope.get("store_ids")
                        or []
                    )
                    if str(item or "").strip()
                ],
                "tenantFilterPolicy": (
                    "The SQL executor automatically binds the published merchantFilterColumn "
                    "to this merchantId. Do not expose it as a business dimension."
                ),
            },
            "topicRouting": session.runtime.routing.model_dump(by_alias=True),
            "topicL0Manifests": manifests,
            "thinRecallCandidates": _thin_recall(session.runtime.recall, limit=8),
            "instructions": (
                "Use Topic/recall only for initial navigation. trustedExecutionScope is authoritative. "
                "Progressively read exact files under /knowledge, then use the typed Grounded tools."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _governed_response(
        session: GroundedDeepAgentSession,
        thread_id: str,
        run_id: str,
    ) -> ChatResponse:
        state = session.runtime
        trace = {
            "harness": {
                "runtime": "grounded_deepagent",
                "threadId": thread_id,
                "runId": run_id,
                "activeAttemptId": state.active_attempt_id,
                "activeGeneration": state.active_generation,
                "legacyFallbackUsed": False,
            }
        }
        if state.clarification is not None:
            return ChatResponse(
                answer=state.clarification.question,
                category_name=state.routing.display_summary(),
                clarification=state.clarification,
                debug_trace=trace,
            )
        if state.answer:
            rows = (
                list(state.run_result.merged_query_bundle.rows)
                if state.run_result is not None
                else []
            )
            tables = (
                list(state.run_result.merged_query_bundle.tables)
                if state.run_result is not None
                else []
            )
            return ChatResponse(
                answer=state.answer,
                category_name=state.routing.display_summary(),
                doris_tables=tables,
                data_rows=rows,
                debug_trace=trace,
            )
        raise RuntimeError(
            "Grounded DeepAgent Core ended without verified answer or typed clarification"
        )


def _thin_recall(bundle: RecallBundle, limit: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in bundle.items:
        metadata = dict(item.metadata or {})
        ref_id = str(metadata.get("semanticRefId") or item.doc_id or "").strip()
        path = str(metadata.get("semanticPath") or metadata.get("path") or "")
        kind = str(metadata.get("semanticKind") or item.source_type or "").upper()
        if ref_id.endswith(":asset") or path.endswith("/asset.json") or kind == "TABLE_ASSET":
            continue
        if not ref_id or ref_id in seen:
            continue
        seen.add(ref_id)
        result.append(
            {
                "refId": ref_id,
                "path": path,
                "kind": kind,
                "topic": item.topic,
                "table": item.table,
                "title": item.title,
                "snippet": item.content[:600],
                "score": float(item.fusion_score or 0.0),
            }
        )
        if len(result) >= max(1, int(limit or 1)):
            break
    return result


def _canonical_progressive_ref(ref_id: str) -> str:
    value = str(ref_id or "").strip()
    if value.startswith("semantic:") and value.endswith(":asset"):
        return value[: -len(":asset")] + ":detail"
    return value


def _deepagent_reader_is_core() -> bool:
    """Fail closed for filesystem reads originating from a subagent graph."""

    try:
        from langgraph.config import get_config

        config = get_config()
    except Exception:
        return False
    configurable = dict(config.get("configurable") or {})
    metadata = dict(config.get("metadata") or {})
    checkpoint_ns = str(
        configurable.get("checkpoint_ns")
        or configurable.get("checkpointNamespace")
        or ""
    ).lower()
    identity_text = json.dumps(
        {"configurable": configurable, "metadata": metadata},
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).lower()
    if "general-purpose" in identity_text or "subagent" in identity_text:
        return False
    # LangGraph nested/subgraph executions receive a checkpoint namespace;
    # the root Core invocation does not. Unknown nested identities never gain
    # executable evidence authority.
    if checkpoint_ns and checkpoint_ns not in {"deepagent", "grounded_deepagent"}:
        return False
    return True
