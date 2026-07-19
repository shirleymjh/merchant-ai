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
from merchant_ai.models import (
    AgentActionTrace,
    AgentDecision,
    ChatContext,
    ChatResponse,
    ConversationMessage,
    KnowledgeRequest,
    KnowledgeRequestType,
)
from merchant_ai.services.assets import (
    build_stable_topic_table_manifest,
    normalize_semantic_path,
    semantic_metric_path,
    semantic_relationship_ref_id,
    semantic_relationship_path,
    semantic_table_detail_ref_id,
    semantic_table_detail_path,
    semantic_table_entry_path,
    semantic_table_section_path,
)
from merchant_ai.services.context_filesystem import ContextPathOutsideRootError, resolve_context_path
from merchant_ai.services.grounded_query_contract import semantic_evidence_calculation_capabilities
from merchant_ai.services.language_policy import load_language_policy
from merchant_ai.services.tool_runtime import tool_runtime_scope


LOGGER = logging.getLogger(__name__)
_KNOWLEDGE_SCOPE: ContextVar[Optional[Any]] = ContextVar("diana_knowledge_scope", default=None)


def semantic_workspace_topics(state: AgentState) -> List[str]:
    """Return seed Topics plus manifests the Core explicitly opened."""

    workspace = state.get("topic_workspace") or {}
    values = [
        *list(workspace.get("topics") or []),
        *list(state.get("semantic_workspace_opened_topics") or []),
    ]
    return list(dict.fromkeys(str(item).strip() for item in values if str(item).strip()))


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
    recall_candidate_fingerprint: str = ""
    core_semantic_evidence: List[Dict[str, Any]] = field(default_factory=list)
    semantic_tool_history: Dict[str, int] = field(default_factory=dict)
    semantic_progress_fingerprint: str = ""
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
    MAX_EVIDENCE_SNIPPET_CHARS = 100_000

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
        # Scope checks and catalog resolution must see the same canonical path.
        # Without this, catalog-compatible aliases such as runtime/topics/...
        # bypassed the Topic gate because _scope_error only recognized topics/.
        return normalize_semantic_path(str(value or "/")).rstrip("/")

    @staticmethod
    def _topic_from_path(path: str) -> str:
        parts = [part for part in str(path or "").strip("/").split("/") if part]
        return parts[1] if len(parts) >= 2 and parts[0] == "topics" else ""

    def _allowed_topics(self) -> List[str]:
        scope = _KNOWLEDGE_SCOPE.get()
        if not scope:
            return []
        state = scope.state if isinstance(scope, _DianaLeadSession) else scope
        return semantic_workspace_topics(state)

    @staticmethod
    def _is_topic_index_path(path: str) -> bool:
        return str(path or "").strip("/") == "topics/index.json"

    @staticmethod
    def _manifest_topic_from_path(path: str) -> str:
        parts = [part for part in str(path or "").strip("/").split("/") if part]
        if len(parts) == 3 and parts[0] == "topics" and parts[2] == "manifest.json":
            return parts[1]
        return ""

    def _record_opened_topic_manifest(self, topic: str) -> None:
        scope = _KNOWLEDGE_SCOPE.get()
        if not scope or not topic:
            return

        def update(state: AgentState) -> None:
            opened = list(state.get("semantic_workspace_opened_topics") or [])
            if topic not in opened and topic not in (state.get("topic_workspace") or {}).get("topics", []):
                opened.append(topic)
            state["semantic_workspace_opened_topics"] = opened
            workspace = dict(state.get("topic_workspace") or {})
            workspace["openedTopics"] = list(opened)
            workspace["effectiveTopics"] = semantic_workspace_topics(state)
            state["topic_workspace"] = workspace

        if isinstance(scope, _DianaLeadSession):
            with scope.lock:
                update(scope.state)
        elif isinstance(scope, dict):
            update(scope)

    @staticmethod
    def _mark_topic_index_read() -> None:
        scope = _KNOWLEDGE_SCOPE.get()
        if isinstance(scope, _DianaLeadSession):
            with scope.lock:
                scope.state["semantic_topic_index_read"] = True
        elif isinstance(scope, dict):
            scope["semantic_topic_index_read"] = True

    @staticmethod
    def _topic_index_was_read() -> bool:
        scope = _KNOWLEDGE_SCOPE.get()
        if not scope:
            return False
        state = scope.state if isinstance(scope, _DianaLeadSession) else scope
        return bool(state.get("semantic_topic_index_read"))

    @staticmethod
    def _topic_expansion_locked() -> bool:
        scope = _KNOWLEDGE_SCOPE.get()
        if not scope:
            return False
        state = scope.state if isinstance(scope, _DianaLeadSession) else scope
        workspace = state.get("topic_workspace") or {}
        return bool(
            workspace.get("isolated")
            or str(workspace.get("mode") or "") == "explicit_topic_scope"
            or str(workspace.get("expansionPolicy") or "") == "user_locked"
        )

    def _scope_error(self, path: str = "") -> str:
        allowed = self._allowed_topics()
        if not allowed:
            return "TOPIC_SCOPE_REQUIRED: route and confirm a Topic before reading /knowledge"
        if self._is_topic_index_path(path):
            return ""
        manifest_topic = self._manifest_topic_from_path(path)
        if manifest_topic:
            if manifest_topic in allowed:
                return ""
            if self._topic_expansion_locked():
                return "TOPIC_SCOPE_LOCKED: the user explicitly restricted this query to the seed Topic"
            if self._topic_index_was_read():
                return ""
            return "TOPIC_INDEX_READ_REQUIRED: read /topics/index.json before opening a non-seed Topic manifest"
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
        complete: bool,
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
            # Seal exactly the bytes stored in the ledger.  The previous code
            # hashed the unbounded content but stored a 4k prefix, causing every
            # larger successful read to fail its own integrity check later.
            "contentHash": hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            "contentComplete": bool(complete and len(excerpt) == len(str(content or ""))),
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

    @staticmethod
    def _semantic_progress(scope: _DianaLeadSession) -> str:
        contract = scope.state.get("grounded_query_contract")
        contract_status = str(
            getattr(contract, "status", "")
            or (contract.get("status") if isinstance(contract, dict) else "")
            or ""
        )
        payload = {
            "actionCount": int(scope.action_count or 0),
            "topics": semantic_workspace_topics(scope.state),
            "topicIndexRead": bool(scope.state.get("semantic_topic_index_read")),
            "contractStatus": contract_status,
            "evidence": [
                (
                    str(item.get("refId") or ""),
                    str(item.get("contentHash") or ""),
                    bool(item.get("contentComplete")),
                )
                for item in scope.core_semantic_evidence
            ],
            "rejected": [
                str(item.get("fingerprint") or "")
                for item in scope.state.get("grounded_rejected_bindings") or []
                if isinstance(item, dict)
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()

    def _semantic_tool_loop_error(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        outcome: str,
    ) -> str:
        """Block identical semantic tool calls after repeated no-progress results."""

        scope = _KNOWLEDGE_SCOPE.get()
        if not isinstance(scope, _DianaLeadSession):
            return ""
        with scope.lock:
            progress = self._semantic_progress(scope)
            if progress != scope.semantic_progress_fingerprint:
                scope.semantic_progress_fingerprint = progress
                scope.semantic_tool_history = {}
            key = hashlib.sha256(
                json.dumps(
                    {
                        "tool": str(tool_name or ""),
                        "arguments": arguments,
                        "outcome": str(outcome or ""),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ).encode("utf-8")
            ).hexdigest()
            count = int(scope.semantic_tool_history.get(key) or 0) + 1
            scope.semantic_tool_history[key] = count
            if count < 4:
                return ""
            scope.state["semantic_tool_loop_guard"] = {
                "code": "SEMANTIC_TOOL_NO_PROGRESS_BLOCKED",
                "tool": tool_name,
                "count": count,
                "arguments": dict(arguments),
                "outcome": str(outcome or "")[:300],
                "progressFingerprint": progress,
            }
            return (
                "SEMANTIC_TOOL_NO_PROGRESS_BLOCKED: identical %s call repeated %d times without "
                "new evidence or binding progress. Do not retry the same call; choose another file, "
                "return to the manifest, or revise bindings."
            ) % (tool_name, count)

    def ls(self, path: str) -> LsResult:
        normalized = self._path(path)
        if not normalized:
            return LsResult(entries=[FileInfo(path="/topics/", is_dir=True, size=0, modified_at="")])
        if normalized == "topics":
            topics = self._allowed_topics()
            if not topics:
                return LsResult(error=self._scope_error(normalized))
            return LsResult(
                entries=[
                    FileInfo(path="/topics/index.json", is_dir=False, size=0, modified_at=""),
                    *[
                        FileInfo(path="/topics/%s/" % topic, is_dir=True, size=0, modified_at="")
                        for topic in topics
                    ],
                ]
            )
        scope_error = self._scope_error(normalized)
        if scope_error:
            return LsResult(error=scope_error)
        topic_match = self._topic_from_path(normalized)
        if normalized.count("/") == 1 and topic_match:
            return LsResult(
                entries=[
                    FileInfo(path="/topics/%s/manifest.json" % topic_match, is_dir=False, size=0, modified_at=""),
                    FileInfo(path="/topics/%s/tables/" % topic_match, is_dir=True, size=0, modified_at=""),
                    FileInfo(path="/topics/%s/relationships.json" % topic_match, is_dir=False, size=0, modified_at=""),
                ]
            )
        tables_match = normalized.split("/")
        if len(tables_match) == 3 and tables_match[0] == "topics" and tables_match[2] == "tables":
            topic = tables_match[1]
            return LsResult(
                entries=[
                    FileInfo(
                        path="/topics/%s/tables/%s/" % (topic, str(item.get("tableName") or "")),
                        is_dir=True,
                        size=0,
                        modified_at="",
                    )
                    for item in self.semantic_catalog.topic_assets.load_manifest(topic)
                    if isinstance(item, dict) and str(item.get("tableName") or "")
                ]
            )
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
            loop_error = self._semantic_tool_loop_error(
                "read_file",
                {"path": normalized, "offset": offset, "limit": limit},
                scope_error,
            )
            return ReadResult(error=loop_error or scope_error)
        try:
            result = self.semantic_catalog.read(path=normalized, max_chars=2_000_000, offset=0)
        except Exception as exc:
            error = "SEMANTIC_READ_FAILED: %s" % str(exc)[:200]
            loop_error = self._semantic_tool_loop_error(
                "read_file",
                {"path": normalized, "offset": offset, "limit": limit},
                error,
            )
            return ReadResult(error=loop_error or error)
        if not result.get("success"):
            error = str(result.get("error") or "SEMANTIC_REF_NOT_FOUND")
            loop_error = self._semantic_tool_loop_error(
                "read_file",
                {"path": normalized, "offset": offset, "limit": limit},
                error,
            )
            return ReadResult(error=loop_error or error)
        kind = str(result.get("kind") or "").upper()
        topic = str(result.get("topic") or "")
        if kind == "TOPIC_INDEX":
            self._mark_topic_index_read()
        allowed_before_open = set(self._allowed_topics())
        if kind == "TOPIC_MANIFEST" and topic and topic not in allowed_before_open:
            if self._topic_expansion_locked():
                error = "TOPIC_SCOPE_LOCKED: the user explicitly restricted this query to the seed Topic"
                loop_error = self._semantic_tool_loop_error(
                    "read_file",
                    {"path": normalized, "offset": offset, "limit": limit},
                    error,
                )
                return ReadResult(error=loop_error or error)
            if not self._topic_index_was_read():
                error = "TOPIC_INDEX_READ_REQUIRED: read /topics/index.json before opening a non-seed Topic manifest"
                loop_error = self._semantic_tool_loop_error(
                    "read_file",
                    {"path": normalized, "offset": offset, "limit": limit},
                    error,
                )
                return ReadResult(error=loop_error or error)
            self._record_opened_topic_manifest(topic)
        elif kind != "TOPIC_INDEX" and topic not in allowed_before_open:
            error = "TOPIC_SCOPE_DENIED: semantic ref escaped the active Topic workspace"
            loop_error = self._semantic_tool_loop_error(
                "read_file",
                {"path": normalized, "offset": offset, "limit": limit},
                error,
            )
            return ReadResult(error=loop_error or error)
        lines = str(result.get("content") or "").splitlines(keepends=True)
        start = max(0, int(offset or 0))
        end = start + max(1, int(limit or 1))
        content = "".join(lines[start:end])
        complete = start == 0 and end >= len(lines) and not bool(result.get("truncated"))
        self._record_core_read_evidence(result, normalized, content, start, limit, complete)
        loop_error = self._semantic_tool_loop_error(
            "read_file",
            {"path": normalized, "offset": offset, "limit": limit},
            "%s:%s" % (
                str(result.get("refId") or normalized),
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            ),
        )
        if loop_error:
            return ReadResult(error=loop_error)
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
        if self._is_topic_index_path(normalized):
            try:
                content = str(self.semantic_catalog.topic_index_ref().get("content") or "")
            except Exception as exc:
                return GrepResult(error="SEMANTIC_GREP_FAILED: %s" % str(exc)[:200])
            needle = str(pattern or "").casefold()
            matches = [
                GrepMatch(path="/topics/index.json", line=index, text=line[:500])
                for index, line in enumerate(content.splitlines(), start=1)
                if needle and needle in line.casefold()
            ]
            self._mark_topic_index_read()
            return GrepResult(matches=matches[:100])
        requested_topic = self._topic_from_path(normalized)
        topics = [requested_topic] if requested_topic else self._allowed_topics()
        catalog_path = "" if normalized in {"", "topics"} else normalized
        hits: List[Dict[str, Any]] = []
        try:
            for topic in topics:
                hits.extend(
                    self.semantic_catalog.grep(
                        query=pattern,
                        topic=topic,
                        limit=100,
                        path=catalog_path,
                    )
                )
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
    # The DeepAgent runtime has one planning authority: GroundedQueryContract.
    # These actions belong to the legacy QuestionUnderstanding/Planner graph or
    # have a dedicated grounded-mode tool.  They must never be reachable through
    # run_diana_action in grounded mode.
    GROUNDED_MODE_BLOCKED_ACTION_IDS = frozenset(
        {
            "fast_understand",
            "try_fast_metric",
            "retrieve_knowledge",
            "compact_assets",
            "query_metric",
            "plan_graph",
            "reflect_plan",
            "validate_graph",
            "repair_graph",
        }
    )
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

    GROUNDING_TABLE_DETAIL_KIND = "TABLE_DETAIL"
    GROUNDING_EXACT_EVIDENCE_KINDS = frozenset(
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
- Topic selection is automatic and internal; never ask a merchant to choose a Topic name. The routing observation separates business ownership from the serving workspace and records governed selection evidence. It then refreshes one Topic-local hybrid recall using the complete user question. The first post-Topic observation contains both the complete L0 table manifest and thin recallCandidates; recall candidates are navigation hints, never planning authority.
- Use native ls/read_file/grep to inspect exact semantic assets, then call commit_grounded_query_contract with the selected refs. Call compile_grounded_query to deterministically produce and validate QueryGraph from that contract.
- fast_understand, compact_assets, query_metric, plan_graph, reflect_plan, validate_graph and repair_graph belong to the legacy QuestionUnderstanding/Planner architecture and are disabled in this runtime. The GroundedQueryContract is the only planning authority.
- Select run_diana_action only for a returned governed execution, evidence, clarification, or answer action. Use its next observation/catalog to decide again; do not assume a fixed order.
- When the observation requires human clarification, immediately run the returned ask_human action. Do not retrieve, read more files, commit a contract or compile while clarification is pending.
- run_diana_action cannot create semantic bindings or plan QueryGraph; those authorities belong to the independent grounded-contract tools.
- Treat every plan as a revisable hypothesis. After freshness gaps, zero rows, SQL errors, or an unsupported dimension, inspect the observation and independently choose whether to read more knowledge, replan, switch tables, repair/requery, or merge verified results; never run a fixed sequence to completion.
- Zero rows and failed queries are execution observations, never evidence that the business value is zero.
- Never invent metrics, SQL, rows, evidence, or a business answer outside governed tools.

FileSystem-as-Context contract:
- /knowledge is read-only and unavailable until the runtime has narrowed the question to a bounded set of candidate Topics.
- Topic discovery is not a final table decision. The first semantic disclosure is only the candidate Topics' L0 table manifests and business summaries. Compare them, choose the table(s) that cover both the requested fact and dimensions, then read the chosen detailPath.
- The routed Topic is a seed workspace, not a permanent closed world. If its L0 manifest and progressively read table files prove insufficient, grep/read /knowledge/topics/index.json, then read one candidate Topic manifest. A Topic becomes searchable only after its manifest has been opened; do not ask retrieve_knowledge to broaden scope automatically.
- Prefer a detail fact table directly when the question asks for a breakdown, ranking, or unsupported dimension. A profile aggregate is optional for cheap/authoritative summaries; detailMetricRef is navigation evidence, never a mandatory profile-then-detail pipeline.
- You are the knowledge-navigation owner. Use Deep Agents' native ls, read_file and grep tools yourself; Planner does not browse or deep-read the knowledge tree.
- Before committing a contract, read every chosen TABLE_DETAIL plus every exact metric, dimension, rule or relationship definition required by the question.
- Bind only semantic objects required to answer the question. For entity rankings, use the governed entity key as groupByRef. Descriptive name/label enrichment is a separate governed stage and is not part of the current Grounded Query Contract.
- When no published metric is needed and an exact field itself is the measure, bind it through fieldAggregations with an allowlisted COUNT or COUNT_DISTINCT operator. Never author a formula or pretend the field is a published business metric.
- A submitted table/metric is provisional. If the Contract returns REVISE_BINDINGS, do not compile it and do not resubmit a rejected table. Follow requiredCapability: first inspect already-read compatible bindings; if none exist, return to the current Topic L0 manifest; if that Topic still has no capable table, read /knowledge/topics/index.json and open one candidate Topic manifest before continuing.
- Treat progressively-read calculationSemantics/usagePolicy as the authority for how a metric or field may be composed across time, grain and joins. Do not infer these rules from names. When a policy declares NOT_COMPOSABLE, EXACT_ONLY, LAST_VALUE, RATIO_OF_SUMS, WEIGHTED_AVERAGE or an alternativeCapability, follow that declaration and reselect bindings when necessary.
- If contract validation reports another semantic gap, continue with ls/read_file/grep or the dedicated Topic-local retrieve_knowledge tool, then commit a revised contract; never ask a Planner to search for you.
- A pending semantic gap never authorizes Topic expansion. Supplemental retrieve_knowledge stays inside the active Topic workspace; use ls/read_file/grep for metrics or fields inside an already selected table.
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
        self._retrieve_knowledge_tool = self._build_retrieve_knowledge_tool()
        self._commit_grounded_contract_tool = self._build_commit_grounded_contract_tool()
        self._compile_grounded_query_tool = self._build_compile_grounded_query_tool()
        model = lead_llm.chat_model() if lead_llm and lead_llm.configured else None
        if model is not None:
            try:
                self.deep_agent_graph = create_deep_agent(
                    model=model,
                    tools=[
                        self._inspect_tool,
                        self._action_tool,
                        self._retrieve_knowledge_tool,
                        self._commit_grounded_contract_tool,
                        self._compile_grounded_query_tool,
                    ],
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
                adapter._sync_core_semantic_evidence(session)
                payload = adapter._prepare_turn(session)
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

    def _build_retrieve_knowledge_tool(self) -> Any:
        adapter = self

        @tool("retrieve_knowledge")
        def retrieve_knowledge(
            query: str,
            reason: str,
            runtime: ToolRuntime[DeepAgentRunContext],
        ) -> str:
            """Run one targeted Topic-scoped supplemental recall for an explicit semantic gap."""

            session = runtime.context.session
            with session.lock:
                state = session.state
                block_reason = adapter._supplemental_recall_block_reason(session)
                if block_reason:
                    session.action_count += 1
                    payload = adapter._prepare_turn(session)
                    payload["supplementalRecallBlocked"] = {
                        "code": "FILESYSTEM_NAVIGATION_REQUIRED",
                        "reason": block_reason,
                        "next": (
                            "Use ls/grep/read_file inside the selected table. If the seed Topic is proven "
                            "insufficient, grep/read /knowledge/topics/index.json and open a candidate manifest."
                        ),
                    }
                    return json.dumps(payload, ensure_ascii=False, default=str)
                request = KnowledgeRequest(
                    type=KnowledgeRequestType.TABLE,
                    query=str(query or "").strip() or str(state.get("question") or ""),
                    reason=str(reason or "Core requested supplemental semantic navigation")[:500],
                    round=int(state.get("query_graph_supplemental_retrieve_count") or 0) + 1,
                )
                state["pending_knowledge_requests"] = [
                    *list(state.get("pending_knowledge_requests") or []),
                    request,
                ]
                state["_core_targeted_topic_recall"] = True
                try:
                    session.state = adapter.domain_workflow.retrieve_knowledge(state)
                finally:
                    session.state.pop("_core_targeted_topic_recall", None)
                session.action_count += 1
                payload = adapter._prepare_turn(session)
            return json.dumps(payload, ensure_ascii=False, default=str)

        return retrieve_knowledge

    def _supplemental_recall_block_reason(self, session: _DianaLeadSession) -> str:
        evidence = self._active_core_semantic_evidence(session)
        if not any(str(item.get("kind") or "").upper() == "TABLE_DETAIL" for item in evidence):
            return ""
        contract = session.state.get("grounded_query_contract")
        gaps = []
        if contract is not None:
            gaps = list(
                getattr(contract, "unresolved_gaps", None)
                or (contract.get("unresolvedGaps") if isinstance(contract, dict) else [])
                or []
            )
        codes = {
            str(getattr(item, "code", "") or (item.get("code") if isinstance(item, dict) else ""))
            for item in gaps
        }
        read_gap_codes = {
            "TABLE_BINDING_REF_NOT_READ",
            "METRIC_BINDING_REF_NOT_READ",
            "FIELD_AGGREGATION_REF_NOT_READ",
            "DIMENSION_BINDING_REF_NOT_READ",
            "GROUP_BY_BINDING_REF_NOT_READ",
            "RELATIONSHIP_BINDING_REF_NOT_READ",
            "METRIC_EVIDENCE_REQUIRED",
            "RANKING_DIMENSION_REQUIRED",
        }
        if codes and codes <= read_gap_codes:
            return "The current Contract gaps require exact file reads, not another retrieval pass"
        return "A table has already been selected; inspect its detail/section indexes and exact definitions first"

    def _build_commit_grounded_contract_tool(self) -> Any:
        adapter = self

        @tool("commit_grounded_query_contract")
        def commit_grounded_query_contract(
            table_refs: List[str],
            metric_refs: List[str],
            dimension_refs: Optional[List[str]],
            group_by_ref: str,
            label_refs: Optional[Dict[str, str]],
            relationship_refs: Optional[List[str]],
            ranking_order: str,
            limit: int,
            analysis_mode: str,
            time_expression: str,
            reason: str,
            runtime: ToolRuntime[DeepAgentRunContext],
            field_aggregations: Optional[List[Dict[str, Any]]] = None,
        ) -> str:
            """Commit Core-selected, already-read semantic refs as a grounded query contract."""

            session = runtime.context.session
            with session.lock:
                if session.state.get("human_clarification_required"):
                    payload = adapter._prepare_turn(session)
                    payload["groundingCommitBlocked"] = {
                        "code": "HUMAN_CLARIFICATION_PENDING",
                        "next": "Run the governed ask_human action before grounding or compilation",
                    }
                    return json.dumps(payload, ensure_ascii=False, default=str)
                adapter._sync_core_semantic_evidence(session)
                active_evidence = adapter._active_core_semantic_evidence(session)
                evidence_by_ref: Dict[str, Dict[str, Any]] = {}
                for item in active_evidence:
                    evidence_ref = str(item.get("refId") or "")
                    if not evidence_ref:
                        continue
                    previous = evidence_by_ref.get(evidence_ref)
                    if previous is None or bool(item.get("contentComplete")) or not bool(previous.get("contentComplete")):
                        evidence_by_ref[evidence_ref] = dict(item)
                available_refs = set(evidence_by_ref)

                def canonical(ref_id: str) -> str:
                    value = str(ref_id or "").strip()
                    if not value or value in available_refs:
                        return value
                    alternates = []
                    if ":column:" in value:
                        alternates.append(value.replace(":column:", ":field:", 1))
                    if ":field:" in value:
                        alternates.append(value.replace(":field:", ":column:", 1))
                    return next((item for item in alternates if item in available_refs), value)

                canonical_table_refs = list(dict.fromkeys(canonical(item) for item in table_refs or [] if canonical(item)))
                canonical_metric_refs = list(dict.fromkeys(canonical(item) for item in metric_refs or [] if canonical(item)))
                canonical_field_aggregations: List[Dict[str, str]] = []
                for raw in field_aggregations or []:
                    if not isinstance(raw, dict):
                        continue
                    field_ref = canonical(str(raw.get("fieldRef") or raw.get("field_ref") or ""))
                    if not field_ref:
                        continue
                    canonical_field_aggregations.append(
                        {
                            "fieldRef": field_ref,
                            "aggregation": str(raw.get("aggregation") or ""),
                            "requestedPhrase": str(
                                raw.get("requestedPhrase")
                                or raw.get("requested_phrase")
                                or ""
                            ),
                        }
                    )
                canonical_dimension_refs = list(dict.fromkeys(canonical(item) for item in dimension_refs or [] if canonical(item)))
                canonical_relationship_refs = list(dict.fromkeys(canonical(item) for item in relationship_refs or [] if canonical(item)))
                canonical_group_by_ref = canonical(group_by_ref)
                expected_ref_kinds = [
                    ("tableRefs", "TABLE_DETAIL", canonical_table_refs),
                    ("metricRefs", "METRIC", canonical_metric_refs),
                    (
                        "fieldAggregations",
                        "COLUMN",
                        [item["fieldRef"] for item in canonical_field_aggregations],
                    ),
                    ("dimensionRefs", "COLUMN", canonical_dimension_refs),
                    ("relationshipRefs", "RELATIONSHIPS", canonical_relationship_refs),
                    ("groupByRef", "COLUMN", [canonical_group_by_ref] if canonical_group_by_ref else []),
                ]
                required_reads: List[Dict[str, Any]] = []
                required_identities: set[tuple[str, str]] = set()
                for binding, expected_kind, refs in expected_ref_kinds:
                    for selected_ref in refs:
                        observed = evidence_by_ref.get(selected_ref, {})
                        observed_kind = str(observed.get("kind") or "").upper()
                        complete = observed.get("contentComplete") is not False
                        if observed_kind == expected_kind and complete:
                            continue
                        identity = (selected_ref, expected_kind)
                        if identity in required_identities:
                            continue
                        required_identities.add(identity)
                        required_reads.append(
                            adapter._grounding_required_read(
                                selected_ref,
                                binding=binding,
                                expected_kind=expected_kind,
                                observed=observed,
                                evidence=active_evidence,
                            )
                        )
                if required_reads:
                    session.action_count += 1
                    payload = adapter._prepare_turn(session)
                    payload["groundingReadRequired"] = {
                        "code": "SELECTED_REFS_NOT_READ_OR_WRONG_KIND",
                        "requiredReads": required_reads,
                        "availableRefs": sorted(available_refs)[-32:],
                        "next": (
                            "Call read_file on each requiredReads[].readPath exactly as provided, then resubmit "
                            "using the returned refId only when returnedKind equals expectedKind. Do not use "
                            "asset.json, section index refs, fragments, or another retrieval pass as bindings."
                        ),
                    }
                    return json.dumps(payload, ensure_ascii=False, default=str)
                rejected_submission = adapter._rejected_submission_block(
                    session.state,
                    canonical_table_refs,
                    canonical_metric_refs,
                    canonical_field_aggregations,
                    evidence_by_ref,
                )
                if rejected_submission:
                    session.action_count += 1
                    payload = adapter._prepare_turn(session)
                    payload["groundingCommitBlocked"] = rejected_submission
                    return json.dumps(payload, ensure_ascii=False, default=str)
                binding_hints = {
                    "tableRefs": canonical_table_refs,
                    "metricRefs": canonical_metric_refs,
                    "fieldAggregations": canonical_field_aggregations,
                    "dimensionRefs": canonical_dimension_refs,
                    "groupByRef": canonical_group_by_ref,
                    "labelRefs": dict(label_refs or {}),
                    "relationshipRefs": canonical_relationship_refs,
                    "ranking": {
                        "order": str(ranking_order or ""),
                        "limit": max(0, int(limit or 0)),
                    },
                    "analysisMode": str(analysis_mode or ""),
                    "timeExpression": str(time_expression or ""),
                    "reason": str(reason or "")[:500],
                }
                session.state = adapter.domain_workflow.commit_grounded_query_contract(
                    session.state,
                    binding_hints,
                )
                session.action_count += 1
                payload = adapter._prepare_turn(session)
            return json.dumps(payload, ensure_ascii=False, default=str)

        return commit_grounded_query_contract

    @staticmethod
    def _grounding_required_read(
        selected_ref: str,
        binding: str,
        expected_kind: str,
        observed: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return an executable filesystem step for a rejected binding ref."""

        value = str(selected_ref or "").strip()
        parts = value.split(":")
        topic = str(observed.get("topic") or "")
        table = str(observed.get("table") or "")
        key = ""
        if len(parts) >= 3 and parts[0] == "semantic":
            topic = topic or parts[1]
            if len(parts) >= 4 and parts[2] not in {"manifest", "relationships"}:
                table = table or parts[2]
            if len(parts) >= 5:
                key = ":".join(parts[4:]).strip()

        if (not topic or (expected_kind != "RELATIONSHIPS" and not table)) and evidence:
            compatible = next(
                (
                    item
                    for item in evidence
                    if str(item.get("kind") or "").upper() == "TABLE_DETAIL"
                    and (not topic or str(item.get("topic") or "") == topic)
                ),
                {},
            )
            topic = topic or str(compatible.get("topic") or "")
            table = table or str(compatible.get("table") or "")

        read_path = ""
        index_path = ""
        if expected_kind == "TABLE_DETAIL" and topic and table:
            read_path = "/knowledge/" + semantic_table_detail_path(topic, table)
        elif expected_kind == "METRIC" and topic and table:
            index_path = "/knowledge/" + semantic_table_section_path(topic, table, "metrics")
            if key and key != "index":
                read_path = "/knowledge/" + semantic_table_entry_path(topic, table, "metrics", key)
        elif expected_kind == "COLUMN" and topic and table:
            index_path = "/knowledge/" + semantic_table_section_path(topic, table, "columns")
            if key and key != "index":
                read_path = "/knowledge/" + semantic_table_entry_path(topic, table, "columns", key)
        elif expected_kind == "RELATIONSHIPS" and topic:
            read_path = "/knowledge/" + semantic_relationship_path(topic)

        if not read_path:
            read_path = index_path
        return {
            "binding": binding,
            "submittedRef": value,
            "expectedKind": expected_kind,
            "observedKind": str(observed.get("kind") or "").upper() or "NOT_READ",
            "readPath": read_path,
            "indexPath": index_path,
            "instruction": (
                "Read the exact definition file"
                if read_path and read_path != index_path
                else "Read this index, choose one entry, then read that exact entry file"
            ),
        }

    @staticmethod
    def _rejected_submission_block(
        state: AgentState,
        table_refs: List[str],
        metric_refs: List[str],
        field_aggregations: List[Dict[str, str]],
        evidence_by_ref: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fail fast when Core resubmits a table already proven insufficient."""

        selected_tables = {
            str((evidence_by_ref.get(ref_id) or {}).get("table") or "")
            for ref_id in table_refs
            if str((evidence_by_ref.get(ref_id) or {}).get("table") or "")
        }
        if not selected_tables:
            return {}

        def satisfies(table: str, capability: Dict[str, Any]) -> bool:
            operation = str(capability.get("operation") or "").upper()
            required_field_role = str(capability.get("requiredFieldRole") or "").upper()
            required_entity_role = str(capability.get("entityRole") or "").upper()

            def semantic_match(ref_id: str, submitted_operation: str = "") -> bool:
                evidence = evidence_by_ref.get(ref_id) or {}
                if str(evidence.get("table") or "") != table:
                    return False
                try:
                    payload = json.loads(str(evidence.get("contentSnippet") or "{}"))
                except (TypeError, ValueError, json.JSONDecodeError):
                    return False
                kind = str(evidence.get("kind") or "").upper()
                available = semantic_evidence_calculation_capabilities(kind, payload)
                supported_operations = {
                    str(item or "").strip().upper()
                    for item in [
                        submitted_operation,
                        *list(available.get("allowedAggregations") or []),
                        available.get("declaredAggregation"),
                        *[
                            measure.get("operation")
                            for measure in available.get("derivableMeasures") or []
                            if isinstance(measure, dict)
                        ],
                    ]
                    if str(item or "").strip()
                }
                if operation and operation not in supported_operations:
                    return False
                definition = payload.get("definition") if isinstance(payload, dict) else {}
                field_role = str(
                    (definition or {}).get("role")
                    or (definition or {}).get("semanticRole")
                    or ""
                ).upper()
                entity_role = str(
                    available.get("semanticEntityRole")
                    or available.get("entityRole")
                    or ""
                ).upper()
                if required_field_role and field_role != required_field_role:
                    return False
                if required_entity_role and entity_role != required_entity_role:
                    return False
                return True

            for item in field_aggregations:
                if semantic_match(
                    str(item.get("fieldRef") or ""),
                    str(item.get("aggregation") or "").upper().replace("-", "_"),
                ):
                    return True
            for ref_id in metric_refs:
                if semantic_match(ref_id):
                    return True
            return False

        for raw in state.get("grounded_rejected_bindings") or []:
            if not isinstance(raw, dict):
                continue
            table = str(raw.get("table") or "")
            if not table or table not in selected_tables:
                continue
            capability = dict(
                raw.get("requiredCapability")
                or raw.get("required_capability")
                or {}
            )
            if satisfies(table, capability):
                continue
            topics = semantic_workspace_topics(state)
            return {
                "code": "REJECTED_BINDING_REUSED",
                "status": "REVISE_BINDINGS",
                "rejectedTable": table,
                "requiredCapability": capability,
                "searchRequired": True,
                "searchSequence": [
                    *[
                        {
                            "stage": "CURRENT_TOPIC_L0",
                            "path": "/knowledge/topics/%s/manifest.json" % topic,
                        }
                        for topic in topics
                    ],
                    {
                        "stage": "TOPIC_INDEX",
                        "path": "/knowledge/topics/index.json",
                    },
                ],
                "next": (
                    "Do not resubmit this table. Inspect the current L0 manifest; if no table can satisfy "
                    "requiredCapability, read the Topic Index and open one candidate Topic manifest."
                ),
            }
        return {}

    def _build_compile_grounded_query_tool(self) -> Any:
        adapter = self

        @tool("compile_grounded_query")
        def compile_grounded_query(
            reason: str,
            runtime: ToolRuntime[DeepAgentRunContext],
        ) -> str:
            """Deterministically compile and validate QueryGraph from the grounded contract."""

            session = runtime.context.session
            with session.lock:
                session.state["grounded_compile_reason"] = str(reason or "")[:500]
                session.state = adapter.domain_workflow.compile_grounded_query(session.state)
                session.action_count += 1
                payload = adapter._prepare_turn(session)
            return json.dumps(payload, ensure_ascii=False, default=str)

        return compile_grounded_query

    @staticmethod
    def _active_core_semantic_evidence(session: _DianaLeadSession) -> List[Dict[str, Any]]:
        """Return trusted reads still inside the current Topic workspace."""

        allowed_topics = set(semantic_workspace_topics(session.state))
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

    def _grounded_binding_evidence_gaps(self, session: _DianaLeadSession) -> List[str]:
        """Describe missing evidence before a grounded binding can be proposed."""

        evidence = self._active_core_semantic_evidence(session)
        table_details = [
            item
            for item in evidence
            if str(item.get("kind") or "").upper() == self.GROUNDING_TABLE_DETAIL_KIND
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
            if kind not in self.GROUNDING_EXACT_EVIDENCE_KINDS:
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
        available = [
            action_id
            for action_id in decision.available_actions
            if action_id not in self.GROUNDED_MODE_BLOCKED_ACTION_IDS
        ]
        state["available_actions"] = self.domain_workflow.policy.registry.actions(available)
        session.state = state
        session.observation = observation
        session.available_actions = tuple(available)
        return self._turn_payload(session)

    def _ensure_initial_topic_recall(self, state: AgentState) -> AgentState:
        """Refresh thin Topic-local candidates before Core starts table selection.

        This is a context bootstrap invariant rather than a model-selected
        workflow node.  It deliberately uses the complete user question, does
        not expand Topic scope, and leaves later targeted recall decisions to
        the Core ReAct loop.
        """

        workspace = state.get("topic_workspace") or {}
        topics = [str(item).strip() for item in workspace.get("topics") or [] if str(item).strip()]
        if (
            not state.get("topic_routed")
            or not topics
            or state.get("human_clarification_required")
            or state.get("initial_topic_recall_completed")
        ):
            return state
        if self._ranking_requires_time_clarification(state):
            self.domain_workflow.request_human_clarification(
                state,
                "这类排行需要明确时间范围，请问要查询最近多久？",
                "QUERY_PLAN",
                "time_window",
                list(load_language_policy().routing.time_clarification_options),
            )
            return state
        state["_initial_topic_recall"] = True
        try:
            return self.domain_workflow.retrieve_knowledge(state)
        finally:
            state.pop("_initial_topic_recall", None)

    @staticmethod
    def _ranking_requires_time_clarification(state: AgentState) -> bool:
        slots = state.get("route_slots")
        if isinstance(slots, dict):
            time_window = slots.get("timeWindow") or slots.get("time_window") or {}
            raw = str(time_window.get("raw") or "") if isinstance(time_window, dict) else ""
            object_refs = list(slots.get("objectRefs") or slots.get("object_refs") or [])
            signals = [str(item) for item in slots.get("analysisSignals") or slots.get("analysis_signals") or []]
        else:
            time_window = getattr(slots, "time_window", None)
            raw = str(getattr(time_window, "raw", "") or "")
            object_refs = list(getattr(slots, "object_refs", None) or [])
            signals = [str(item) for item in getattr(slots, "analysis_signals", None) or []]
        clarification = state.get("clarification_resolution") or {}
        clarified_days = int(clarification.get("timeWindowDays") or clarification.get("time_window_days") or 0)
        return (
            not raw.strip()
            and clarified_days <= 0
            and not object_refs
            and "typed_ranking_span" in signals
        )

    @staticmethod
    def _thin_recall_candidates(state: AgentState, limit: int = 12) -> List[Dict[str, Any]]:
        """Return navigation-only refs; never expose a planning asset bundle."""

        allowed_topics = set(semantic_workspace_topics(state))
        candidates: List[Dict[str, Any]] = []
        seen_refs: set[str] = set()
        bundle = state.get("recall_bundle")
        for item in list(getattr(bundle, "items", None) or []):
            topic = str(getattr(item, "topic", "") or "").strip()
            if allowed_topics and topic and topic not in allowed_topics:
                continue
            metadata = dict(getattr(item, "metadata", None) or {})
            source_type = str(getattr(item, "source_type", "") or "").upper()
            if source_type == "GOVERNED_RULE":
                # Rule chunks currently arrive as inline retrieval evidence,
                # not as files in the semantic virtual tree.  Never leak their
                # host doc_id or pretend they are readable navigation refs.
                continue
            ref_id = str(metadata.get("semanticRefId") or getattr(item, "doc_id", "") or "").strip()
            path = str(metadata.get("semanticPath") or "").strip()
            table = str(getattr(item, "table", "") or metadata.get("tableName") or "").strip()
            kind = str(metadata.get("semanticKind") or getattr(item, "source_type", "") or "").upper()
            snippet = str(getattr(item, "content", "") or "")[:600]

            # Legacy table-asset recall is useful only as a table candidacy
            # signal.  Convert it to the L1 detail coordinate so Core cannot
            # bypass progressive disclosure or receive formula/schema dumps.
            if kind in {"TABLE_ASSET", "SEMANTIC_TABLE_ASSET"} and topic and table:
                ref_id = semantic_table_detail_ref_id(topic, table)
                path = semantic_table_detail_path(topic, table)
                kind = "TABLE_DETAIL"
                snippet = str(getattr(item, "title", "") or table)[:240]
            elif kind in {"METRIC", "SEMANTIC_METRIC"} and topic and table:
                metric_key = str(metadata.get("metricKey") or "").strip()
                if not metric_key and ":metric:" in ref_id:
                    metric_key = ref_id.split(":metric:", 1)[1]
                if metric_key:
                    path = semantic_metric_path(topic, table, metric_key)
                    kind = "METRIC"
            elif kind in {"RELATIONSHIP", "RELATIONSHIPS", "SEMANTIC_RELATIONSHIP"} and topic:
                ref_id = semantic_relationship_ref_id(topic)
                path = semantic_relationship_path(topic)
                kind = "RELATIONSHIPS"
            if not ref_id.startswith("semantic:"):
                continue
            if not path or ref_id in seen_refs:
                continue
            seen_refs.add(ref_id)
            candidates.append(
                {
                    "refId": ref_id,
                    "path": path,
                    "kind": kind,
                    "topic": topic,
                    "table": table,
                    "title": str(getattr(item, "title", "") or "")[:240],
                    "snippet": snippet,
                    "score": float(getattr(item, "fusion_score", 0.0) or 0.0),
                    "authority": "navigation_candidate_only",
                }
            )
            if len(candidates) >= max(1, int(limit or 1)):
                break
        return candidates

    def _semantic_evidence_ledger(self, session: _DianaLeadSession) -> List[Dict[str, Any]]:
        """Expose every observation with an explicit epistemic status.

        L0 and recall are valid navigation evidence, but only exact Core reads
        may become executable semantic bindings.  Contract and validation
        transitions upgrade the same refs instead of pretending every item has
        equal authority.
        """

        state = session.state
        ledger: List[Dict[str, Any]] = []
        topics = semantic_workspace_topics(state)
        topic_assets = getattr(getattr(self, "semantic_catalog", None), "topic_assets", None)
        if topics and topic_assets is not None:
            manifest = build_stable_topic_table_manifest(topic_assets, topics)
            for table in manifest.get("tables") or []:
                ledger.append(
                    {
                        "refId": str(table.get("detailRefId") or ""),
                        "kind": "TABLE_DETAIL",
                        "topic": str(table.get("topic") or ""),
                        "table": str(table.get("table") or ""),
                        "status": "catalog",
                        "usableFor": ["navigation", "table_candidacy"],
                    }
                )
        for item in self._thin_recall_candidates(state, limit=24):
            ledger.append(
                {
                    "refId": item.get("refId", ""),
                    "kind": item.get("kind", ""),
                    "topic": item.get("topic", ""),
                    "table": item.get("table", ""),
                    "status": "candidate",
                    "usableFor": ["navigation"],
                }
            )
        bound_refs = set()
        contract = state.get("grounded_query_contract")
        if contract is not None:
            contract_status = str(
                getattr(contract, "status", "")
                or (contract.get("status") if isinstance(contract, dict) else "")
                or ""
            )
            raw_ready = (
                getattr(contract, "ready", None)
                if not isinstance(contract, dict)
                else contract.get("ready")
            )
            contract_ready = contract_status == "READY" if raw_ready is None else bool(raw_ready)
            if contract_status == "READY" and contract_ready:
                bound_refs = {
                    str(item)
                    for item in (getattr(contract, "evidence_refs", None) or (contract.get("evidenceRefs") if isinstance(contract, dict) else []) or [])
                    if str(item)
                }
        rejected_by_ref: Dict[str, Dict[str, Any]] = {}
        for raw in state.get("grounded_rejected_bindings") or []:
            if not isinstance(raw, dict):
                continue
            for ref_id in raw.get("refIds") or raw.get("ref_ids") or []:
                if str(ref_id):
                    rejected_by_ref[str(ref_id)] = raw
        for item in self._active_core_semantic_evidence(session):
            ref_id = str(item.get("refId") or "")
            kind = str(item.get("kind") or "").upper()
            complete = item.get("contentComplete") is not False
            bindable_kinds = {"TABLE_DETAIL", "METRIC", "COLUMN", "RELATIONSHIPS"}
            navigation_kinds = {
                "TOPIC_MANIFEST",
                "METRIC_CATALOG",
                "COLUMN_DETAILS",
                "TERMINOLOGY",
                "BUSINESS_RULES",
            }
            if ref_id in bound_refs:
                status = "bound"
                usable_for = ["grounded_contract", "query_compilation"]
            elif ref_id in rejected_by_ref:
                status = "rejected_binding"
                usable_for = ["diagnostic", "do_not_reuse"]
            elif not complete:
                status = "partial_read"
                usable_for = ["navigation", "continue_read"]
            elif kind in bindable_kinds:
                status = "grounded"
                usable_for = ["semantic_reasoning", "grounded_contract_candidate"]
            elif kind in navigation_kinds:
                status = "read_navigation"
                usable_for = ["navigation"]
            else:
                status = "read_evidence"
                usable_for = ["semantic_reasoning"]
            ledger.append(
                {
                    "refId": ref_id,
                    "kind": kind,
                    "topic": str(item.get("topic") or ""),
                    "table": str(item.get("table") or ""),
                    "path": str(item.get("path") or ""),
                    "status": status,
                    "usableFor": usable_for,
                    "contentComplete": complete,
                    "contentHash": str(item.get("contentHash") or ""),
                    **(
                        {
                            "rejectionCode": str(rejected_by_ref[ref_id].get("code") or ""),
                            "requiredCapability": dict(
                                rejected_by_ref[ref_id].get("requiredCapability")
                                or rejected_by_ref[ref_id].get("required_capability")
                                or {}
                            ),
                        }
                        if ref_id in rejected_by_ref
                        else {}
                    ),
                }
            )
        deduped: Dict[tuple[str, str], Dict[str, Any]] = {}
        status_rank = {
            "catalog": 0,
            "candidate": 1,
            "partial_read": 2,
            "read_navigation": 3,
            "read_evidence": 3,
            "grounded": 4,
            "rejected_binding": 5,
            "bound": 6,
        }
        for item in ledger:
            identity = (str(item.get("refId") or ""), str(item.get("status") or ""))
            existing = next(
                (
                    value
                    for (ref_id, _), value in deduped.items()
                    if ref_id == identity[0]
                ),
                None,
            )
            if existing is not None and status_rank.get(str(existing.get("status") or ""), 0) >= status_rank.get(identity[1], 0):
                continue
            for key in [key for key in deduped if key[0] == identity[0]]:
                deduped.pop(key, None)
            deduped[identity] = item
        result = list(deduped.values())[-96:]
        state["semantic_evidence_ledger"] = result
        return result

    @staticmethod
    def _revision_candidate_refs(
        core_evidence: List[Dict[str, Any]],
        required_capabilities: List[Dict[str, Any]],
        rejected_bindings: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        rejected_tables = {
            str(item.get("table") or "")
            for item in rejected_bindings
            if isinstance(item, dict) and str(item.get("table") or "")
        }
        rejected_refs = {
            str(ref_id)
            for item in rejected_bindings
            if isinstance(item, dict)
            for ref_id in item.get("refIds") or item.get("ref_ids") or []
            if str(ref_id)
        }
        capabilities = [item for item in required_capabilities if isinstance(item, dict)]
        candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for item in core_evidence:
            ref_id = str(item.get("refId") or "")
            table = str(item.get("table") or "")
            kind = str(item.get("kind") or "").upper()
            if not ref_id or ref_id in rejected_refs or table in rejected_tables:
                continue
            try:
                payload = json.loads(str(item.get("contentSnippet") or "{}"))
            except (TypeError, ValueError, json.JSONDecodeError):
                payload = {}
            available = semantic_evidence_calculation_capabilities(kind, payload)
            supported_operations = {
                str(operation or "").strip().upper()
                for operation in [
                    *list(available.get("allowedAggregations") or []),
                    available.get("declaredAggregation"),
                    *[
                        measure.get("operation")
                        for measure in available.get("derivableMeasures") or []
                        if isinstance(measure, dict)
                    ],
                ]
                if str(operation or "").strip()
            }
            definition = payload.get("definition") if isinstance(payload, dict) else {}
            declared_field_role = str(
                (definition or {}).get("role")
                or (definition or {}).get("semanticRole")
                or ""
            ).upper()
            declared_entity_role = str(
                available.get("semanticEntityRole")
                or available.get("entityRole")
                or ""
            ).upper()
            compatible = False
            for capability in capabilities:
                operation = str(capability.get("operation") or "").upper()
                if operation and operation not in supported_operations:
                    continue
                required_field_role = str(capability.get("requiredFieldRole") or "").upper()
                if required_field_role and declared_field_role != required_field_role:
                    continue
                required_entity_role = str(capability.get("entityRole") or "").upper()
                if required_entity_role and declared_entity_role != required_entity_role:
                    continue
                compatible = True
                if compatible:
                    break
            if not compatible or ref_id in seen:
                continue
            seen.add(ref_id)
            candidates.append(
                {
                    "refId": ref_id,
                    "path": str(item.get("path") or ""),
                    "kind": kind,
                    "topic": str(item.get("topic") or ""),
                    "table": table,
                }
            )
        return candidates[:16]

    def _binding_revision_payload(
        self,
        session: _DianaLeadSession,
        contract_dump: Dict[str, Any],
        core_evidence: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        gaps = [item for item in contract_dump.get("unresolvedGaps") or [] if isinstance(item, dict)]
        required_capabilities = [
            dict(item.get("requiredCapability") or {})
            for item in gaps
            if item.get("requiredCapability")
        ]
        rejected_bindings = [
            dict(item)
            for item in (
                contract_dump.get("rejectedBindings")
                or session.state.get("grounded_rejected_bindings")
                or []
            )
            if isinstance(item, dict)
        ]
        candidates = self._revision_candidate_refs(
            core_evidence,
            required_capabilities,
            rejected_bindings,
        )
        topics = semantic_workspace_topics(session.state)
        manifest_paths = ["/knowledge/topics/%s/manifest.json" % topic for topic in topics]
        return {
            "status": "REVISE_BINDINGS",
            "rejectedTables": list(
                dict.fromkeys(
                    str(item.get("table") or "")
                    for item in rejected_bindings
                    if str(item.get("table") or "")
                )
            ),
            "requiredCapabilities": required_capabilities,
            "compatibleReadBindings": candidates,
            "searchRequired": not bool(candidates),
            "searchSequence": [
                {
                    "stage": "READ_BINDINGS",
                    "action": "reuse one compatibleReadBindings ref if present",
                },
                {
                    "stage": "CURRENT_TOPIC_L0",
                    "paths": manifest_paths,
                    "action": "choose a non-rejected table and read its detail/section indexes",
                },
                {
                    "stage": "TOPIC_INDEX",
                    "path": "/knowledge/topics/index.json",
                    "action": "only if current Topic manifests have no capable table, open one new Topic manifest",
                },
            ],
            "next": (
                "Recommit with one compatible already-read binding; do not reuse a rejected table"
                if candidates
                else (
                    "No compatible binding is currently read. Return to the current L0 table manifest; "
                    "if it still has no table with the required capability, read the Topic Index and "
                    "open one candidate Topic manifest. Then read exact files and recommit."
                )
            ),
        }

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
            "policy": (
                "actionCatalog contains only governed execution/evidence/answer actions; "
                "semantic navigation and QueryGraph compilation use independent tools. "
                "Catalog order is not a workflow."
            ),
        }
        core_evidence = self._active_core_semantic_evidence(session)
        evidence_gaps = self._grounded_binding_evidence_gaps(session)
        payload["planningAuthority"] = {
            "mode": "grounded_query_contract",
            "legacyPlanningDisabled": True,
            "candidateOwner": "deepagent_core",
            "canonicalizer": "grounded_query_contract_builder",
            "compiler": "grounded_contract_direct_compiler",
            "policy": (
                "Core proposes already-read semantic bindings; Contract canonicalization is fail-closed; "
                "a READY Contract is immutable planning authority and compilation may not infer new bindings."
            ),
        }
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
            "contractProposalReady": not evidence_gaps,
            "missingForContractProposal": evidence_gaps,
        }
        contract = state.get("grounded_query_contract")
        if contract is not None:
            contract_dump = contract.model_dump(by_alias=True) if hasattr(contract, "model_dump") else dict(contract)
            payload["groundedQueryContract"] = {
                "status": str(contract_dump.get("status") or "UNRESOLVED"),
                "ready": bool(contract_dump.get("ready")),
                "executionShape": str(contract_dump.get("executionShape") or ""),
                "primaryTable": str(contract_dump.get("primaryTable") or ""),
                "metricRefs": [
                    str(item.get("semanticRefId") or "")
                    for item in contract_dump.get("metrics") or []
                    if isinstance(item, dict)
                ],
                "dimensionRefs": [
                    str(item.get("semanticRefId") or "")
                    for item in contract_dump.get("dimensions") or []
                    if isinstance(item, dict)
                ],
                "gaps": list(contract_dump.get("unresolvedGaps") or [])[:12],
                "rejectedBindings": list(contract_dump.get("rejectedBindings") or [])[:16],
            }
            if str(contract_dump.get("status") or "") == "REVISE_BINDINGS":
                payload["bindingRevision"] = self._binding_revision_payload(
                    session,
                    contract_dump,
                    core_evidence,
                )
        contract_attempt = state.get("grounded_query_contract_attempt")
        if contract_attempt is not None and contract_attempt is not contract:
            attempt_dump = (
                contract_attempt.model_dump(by_alias=True)
                if hasattr(contract_attempt, "model_dump")
                else dict(contract_attempt)
            )
            active_dump = (
                contract.model_dump(by_alias=True)
                if contract is not None and hasattr(contract, "model_dump")
                else dict(contract)
                if isinstance(contract, dict)
                else {}
            )
            if attempt_dump != active_dump:
                payload["groundedQueryContractAttempt"] = {
                    "status": str(attempt_dump.get("status") or "UNRESOLVED"),
                    "accepted": False,
                    "activeContractPreserved": bool(active_dump.get("ready")),
                    "gaps": list(attempt_dump.get("unresolvedGaps") or [])[:12],
                    "next": (
                        "Complete the candidate using exact read_file evidence and resubmit. "
                        "The last READY Contract remains authoritative until an atomic replacement succeeds."
                    ),
                }
        ledger = self._semantic_evidence_ledger(session)
        payload["semanticEvidenceLedger"] = {
            "entries": ledger,
            "policy": "catalog/candidate guide navigation; grounded/bound refs may compile QueryGraph",
        }
        seed_topics = [str(item) for item in (state.get("topic_workspace") or {}).get("topics") or [] if str(item)]
        topics = semantic_workspace_topics(state)
        workspace_mode = str((state.get("topic_workspace") or {}).get("mode") or "")
        topic_decision = state.get("topic_routing_decision")
        if topic_decision is not None:
            decision_dump = (
                topic_decision.model_dump(by_alias=True)
                if hasattr(topic_decision, "model_dump")
                else dict(topic_decision)
            )
            payload["topicSelection"] = {
                "selectionMode": str(decision_dump.get("selectionMode") or "automatic"),
                "seedTopics": seed_topics,
                "businessTopics": list(
                    (decision_dump.get("selectionEvidence") or {}).get("businessTopics") or []
                ),
                "servingTopics": list(
                    (decision_dump.get("selectionEvidence") or {}).get("servingTopics") or []
                ),
                "queryShape": str(
                    (decision_dump.get("selectionEvidence") or {}).get("queryShape") or ""
                ),
                "sameTableSummaryCandidate": bool(
                    (decision_dump.get("selectionEvidence") or {}).get("sameTableSummaryCandidate")
                ),
                "matchedMetrics": list(
                    (decision_dump.get("selectionEvidence") or {}).get("matchedMetrics") or []
                )[:12],
                "reason": str(decision_dump.get("reason") or ""),
                "userChoiceRequired": False,
            }
        if topics:
            payload["knowledgeRoots"] = [
                "/knowledge/topics/index.json",
                *["/knowledge/topics/%s/manifest.json" % topic for topic in topics],
            ]
            payload["semanticWorkspace"] = {
                "seedTopics": seed_topics,
                "openedTopics": list(state.get("semantic_workspace_opened_topics") or []),
                "effectiveTopics": topics,
                "policy": (
                    "If seed Topics do not cover the question, grep/read the global Topic Index, "
                    "then read one candidate manifest before searching that Topic."
                ),
            }
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
            seed_topics
            and workspace_mode != "clarification_required"
            and not session.table_manifest_disclosed
        ):
            payload["tableManifest"] = build_stable_topic_table_manifest(
                self.semantic_catalog.topic_assets,
                seed_topics,
            )
            payload["semanticDisclosure"] = {
                "layer": "L0",
                "contains": ["topic", "table", "title", "businessSummary", "detailRefId", "detailPath"],
                "omits": ["metrics", "columns", "schema", "rules", "relationships"],
                "next": "choose a table, then read /knowledge/<detailPath>",
            }
            session.table_manifest_disclosed = True
        recall_candidates = self._thin_recall_candidates(state)
        recall_fingerprint = hashlib.sha256(
            json.dumps(
                [(item.get("refId"), item.get("score")) for item in recall_candidates],
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest() if recall_candidates else ""
        if recall_candidates and recall_fingerprint != session.recall_candidate_fingerprint:
            payload["recallCandidates"] = recall_candidates
            payload["recallDisclosure"] = {
                "query": str((state.get("initial_topic_recall_trace") or {}).get("query") or state.get("question") or ""),
                "scope": "active_topic_only",
                "authority": "navigation_only",
                "next": "compare with the full L0 manifest, then read exact semantic files",
            }
            session.recall_candidate_fingerprint = recall_fingerprint
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
        if selected_id in self.GROUNDED_MODE_BLOCKED_ACTION_IDS:
            payload = self._turn_payload(session)
            payload.update(
                {
                    "status": "ACTION_REJECTED",
                    "error": "GROUNDED_MODE_ACTION_DISABLED",
                    "rejectedActionId": selected_id,
                    "next": (
                        "Use successful Core semantic reads to commit_grounded_query_contract, "
                        "then compile_grounded_query."
                    ),
                }
            )
            return payload
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
        if decision.selected_action == "route_topic":
            session.state = self._ensure_initial_topic_recall(session.state)
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
        state["planning_authority"] = "grounded_query_contract"
        state["legacy_planning_disabled"] = True
        state["core_managed_filesystem"] = True
        state = self.domain_workflow.preflight_route(state)
        if self.domain_workflow.preflight_needs_full_context(state):
            state = self.domain_workflow.inherit_context(state)
            state = self.domain_workflow.runtime_bootstrap(state)
        session = _DianaLeadSession(state=state, sink=sink)
        self._prepare_turn(session)
        return session

    def _finish_without_model(self, session: _DianaLeadSession) -> None:
        """Finish without turning an internal runtime failure into clarification."""

        safety_budget = int(self.domain_workflow.policy.max_main_actions) + 4
        while not session.terminal and session.action_count < safety_budget:
            decision = self.domain_workflow.policy.decide(session.state)
            available = [
                action_id
                for action_id in decision.available_actions
                if action_id not in self.GROUNDED_MODE_BLOCKED_ACTION_IDS
            ]
            if not available:
                break
            session.available_actions = tuple(available)
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
        if state.get("human_clarification_required"):
            session.available_actions = ("ask_human",)
            self._execute_action(
                session,
                "ask_human",
                "explicit governed clarification remained pending after DeepAgent stopped",
                decision_source="runtime_fail_closed",
            )
            return

        failure = {
            "code": "GROUNDED_RUNTIME_INCOMPLETE",
            "reason": "DeepAgent stopped before the grounded query reached a governed terminal state",
            "planningAuthority": "grounded_query_contract",
            "legacyFallbackUsed": False,
        }
        state["grounded_runtime_failure"] = failure
        state["partial_answer_reason"] = failure["reason"]
        state.setdefault("degraded_reasons", []).append(failure["code"])
        state.setdefault("runtime_guard_gaps", []).append(failure)
        session.available_actions = ("answer_data",)
        self._execute_action(
            session,
            "answer_data",
            "surface an explicit grounded runtime failure without requesting irrelevant user input",
            decision_source="runtime_fail_closed",
        )
        if not session.terminal and "cache_answer" in session.available_actions:
            self._execute_action(
                session,
                "cache_answer",
                "close the explicit grounded runtime failure response",
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
