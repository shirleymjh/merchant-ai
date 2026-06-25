from __future__ import annotations

import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[2] / "python_backend" / ".env"


class Settings(BaseSettings):
    """Runtime configuration for the Python-only merchant AI backend."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")

    server_port: int = Field(8088, validation_alias="SERVER_PORT")
    company_name: str = Field("yshopping", validation_alias="YSHOPPING_COMPANY_NAME")
    merchant_id: str = Field("100", validation_alias="YSHOPPING_MERCHANT_ID")
    ops_token: str = Field("", validation_alias="YSHOPPING_OPS_TOKEN")

    llm_base_url: str = Field("https://api.openai.com/v1", validation_alias="YSHOPPING_LLM_BASE_URL")
    llm_model: str = Field("gpt-5.2", validation_alias="YSHOPPING_LLM_MODEL")
    llm_api_key: str = Field("", validation_alias="YSHOPPING_LLM_API_KEY")
    llm_request_timeout_seconds: int = Field(20, validation_alias="YSHOPPING_LLM_REQUEST_TIMEOUT_SECONDS")
    llm_planner_timeout_seconds: int = Field(50, validation_alias="YSHOPPING_LLM_PLANNER_TIMEOUT_SECONDS")
    llm_answer_timeout_seconds: int = Field(30, validation_alias="YSHOPPING_LLM_ANSWER_TIMEOUT_SECONDS")
    llm_analysis_timeout_seconds: int = Field(20, validation_alias="YSHOPPING_LLM_ANALYSIS_TIMEOUT_SECONDS")
    llm_max_tokens: int = Field(2048, validation_alias="YSHOPPING_LLM_MAX_TOKENS")
    answer_skill_match_mode: str = Field("off", validation_alias="YSHOPPING_ANSWER_SKILL_MATCH_MODE")

    embedding_base_url: str = Field("https://api.openai.com/v1", validation_alias="YSHOPPING_EMBEDDING_BASE_URL")
    embedding_model: str = Field("text-embedding-3-small", validation_alias="YSHOPPING_EMBEDDING_MODEL")
    embedding_api_key: str = Field("", validation_alias="YSHOPPING_EMBEDDING_API_KEY")
    embedding_dims: int = Field(1536, validation_alias="YSHOPPING_EMBEDDING_DIMS")

    es_enabled: bool = Field(True, validation_alias="YSHOPPING_ES_ENABLED")
    es_base_url: str = Field("http://127.0.0.1:9200", validation_alias="YSHOPPING_ES_BASE_URL")
    es_index: str = Field("merchant_ai_recall", validation_alias="YSHOPPING_ES_INDEX")
    es_api_key: str = Field("", validation_alias="YSHOPPING_ES_API_KEY")
    es_username: str = Field("", validation_alias="YSHOPPING_ES_USERNAME")
    es_password: str = Field("", validation_alias="YSHOPPING_ES_PASSWORD")

    doris_jdbc_url: str = Field(
        "jdbc:mysql://127.0.0.1:9030/yshopping?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai",
        validation_alias="YSHOPPING_DORIS_JDBC_URL",
    )
    doris_username: str = Field("root", validation_alias="YSHOPPING_DORIS_USERNAME")
    doris_password: str = Field("", validation_alias="YSHOPPING_DORIS_PASSWORD")
    doris_read_timeout_seconds: int = Field(30, validation_alias="YSHOPPING_DORIS_READ_TIMEOUT_SECONDS")

    answer_jdbc_url: str = Field(
        "jdbc:mysql://127.0.0.1:9030/yshopping?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai",
        validation_alias="YSHOPPING_ANSWER_JDBC_URL",
    )
    answer_username: str = Field("root", validation_alias="YSHOPPING_ANSWER_USERNAME")
    answer_password: str = Field("", validation_alias="YSHOPPING_ANSWER_PASSWORD")

    harness_workspace_path: str = Field("", validation_alias="YSHOPPING_HARNESS_WORKSPACE")
    context_window_tokens: int = Field(16000, validation_alias="YSHOPPING_HARNESS_CONTEXT_WINDOW_TOKENS")
    tool_result_preview_rows: int = Field(20, validation_alias="YSHOPPING_HARNESS_TOOL_RESULT_PREVIEW_ROWS")
    max_sub_agent_tasks: int = Field(3, validation_alias="YSHOPPING_HARNESS_MAX_SUB_AGENT_TASKS")
    max_concurrent_sub_agents: int = Field(3, validation_alias="YSHOPPING_HARNESS_MAX_CONCURRENT_SUB_AGENTS")
    max_sub_agent_rounds: int = Field(6, validation_alias="YSHOPPING_HARNESS_MAX_SUB_AGENT_ROUNDS")
    agent_v2_enabled: bool = Field(True, validation_alias="YSHOPPING_HARNESS_AGENT_V2_ENABLED")
    agent_mode: str = Field("harness", validation_alias="YSHOPPING_AGENT_MODE")
    agent_node_timeout_seconds: int = Field(45, validation_alias="YSHOPPING_AGENT_NODE_TIMEOUT_SECONDS")
    agent_sql_repair_rounds: int = Field(2, validation_alias="YSHOPPING_AGENT_SQL_REPAIR_ROUNDS")
    agent_max_entity_values: int = Field(200, validation_alias="YSHOPPING_AGENT_MAX_ENTITY_VALUES")
    agent_trace_replay_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_TRACE_REPLAY_ENABLED")
    agent_checkpointer_backend: str = Field("sqlite", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_BACKEND")
    agent_checkpointer_sqlite_path: str = Field("", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_SQLITE_PATH")
    agent_checkpointer_postgres_uri: str = Field("", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_POSTGRES_URI")
    agent_main_rounds: int = Field(18, validation_alias="YSHOPPING_AGENT_MAIN_ROUNDS")
    agent_retrieve_rounds: int = Field(3, validation_alias="YSHOPPING_AGENT_RETRIEVE_ROUNDS")
    agent_plan_rounds: int = Field(1, validation_alias="YSHOPPING_AGENT_PLAN_ROUNDS")
    agent_graph_repair_rounds: int = Field(2, validation_alias="YSHOPPING_AGENT_GRAPH_REPAIR_ROUNDS")
    agent_planner_tool_rounds: int = Field(3, validation_alias="YSHOPPING_AGENT_PLANNER_TOOL_ROUNDS")
    agent_planner_seed_table_limit: int = Field(4, validation_alias="YSHOPPING_AGENT_PLANNER_SEED_TABLE_LIMIT")
    agent_planner_seed_metric_limit: int = Field(14, validation_alias="YSHOPPING_AGENT_PLANNER_SEED_METRIC_LIMIT")
    agent_asset_field_entry_limit: int = Field(240, validation_alias="YSHOPPING_AGENT_ASSET_FIELD_ENTRY_LIMIT")
    agent_planner_prompt_budget_chars: int = Field(14000, validation_alias="YSHOPPING_AGENT_PLANNER_PROMPT_BUDGET_CHARS")
    route_llm_mode: str = Field("low_confidence", validation_alias="YSHOPPING_ROUTE_LLM_MODE")
    context_file_inline_max_chars: int = Field(12000, validation_alias="YSHOPPING_AGENT_CONTEXT_FILE_INLINE_MAX_CHARS")
    context_artifact_inline_max_rows: int = Field(20, validation_alias="YSHOPPING_AGENT_CONTEXT_ARTIFACT_INLINE_MAX_ROWS")

    cache_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_CACHE_ENABLED")
    cache_memory_max_entries: int = Field(512, validation_alias="YSHOPPING_AGENT_CACHE_MEMORY_MAX_ENTRIES")
    cache_doris_select_ttl_seconds: int = Field(60, validation_alias="YSHOPPING_AGENT_CACHE_DORIS_SELECT_TTL_SECONDS")
    cache_recall_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_RECALL_TTL_SECONDS")
    cache_asset_pack_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_ASSET_PACK_TTL_SECONDS")
    cache_llm_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_LLM_TTL_SECONDS")
    python_executable: str = Field(default_factory=lambda: sys.executable, validation_alias="YSHOPPING_PYTHON_EXECUTABLE")

    wiki_path: str = Field("", validation_alias="YSHOPPING_WIKI_PATH")
    topic_path: str = Field("", validation_alias="YSHOPPING_TOPIC_PATH")

    langsmith_tracing: bool = Field(False, validation_alias="LANGSMITH_TRACING")
    langfuse_tracing: bool = Field(False, validation_alias="LANGFUSE_TRACING")

    @property
    def repo_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def resources_root(self) -> Path:
        return self.repo_root / "python_backend" / "resources"

    @property
    def resolved_wiki_path(self) -> Path:
        if self.wiki_path:
            return Path(self.wiki_path)
        return self.resources_root / "wiki"

    @property
    def resolved_topic_path(self) -> Path:
        if self.topic_path:
            return Path(self.topic_path)
        return self.resources_root / "runtime" / "topics"

    @property
    def resolved_sql_path(self) -> Path:
        return self.resources_root / "sql"

    @property
    def resolved_schema_path(self) -> Path:
        return self.resources_root / "schema"

    @property
    def resolved_ops_path(self) -> Path:
        return self.resources_root / "runtime" / "ops"

    @property
    def resolved_workspace_path(self) -> Path:
        if self.harness_workspace_path:
            return Path(self.harness_workspace_path)
        return self.repo_root / "python_backend" / ".merchant-ai"

    @property
    def resolved_checkpointer_sqlite_path(self) -> Path:
        if self.agent_checkpointer_sqlite_path:
            return Path(self.agent_checkpointer_sqlite_path)
        return self.resolved_workspace_path / "checkpoints" / "langgraph.sqlite"

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY") or self.llm_api_key

    @property
    def openai_base_url(self) -> str:
        return os.getenv("OPENAI_BASE_URL") or self.llm_base_url

    @property
    def openai_model(self) -> str:
        return os.getenv("OPENAI_MODEL") or self.llm_model


def jdbc_to_pymysql_kwargs(jdbc_url: str, username: str, password: str) -> Dict[str, object]:
    """Convert a MySQL JDBC URL into PyMySQL connection kwargs."""

    raw = (jdbc_url or "").strip()
    if raw.startswith("jdbc:"):
        raw = raw[len("jdbc:") :]
    parsed = urlparse(raw)
    database = parsed.path[1:] if parsed.path.startswith("/") else parsed.path
    query = parse_qs(parsed.query or "")
    charset = query.get("characterEncoding", ["utf8mb4"])[0]
    return {
        "host": parsed.hostname or "127.0.0.1",
        "port": parsed.port or 3306,
        "user": username,
        "password": password,
        "database": database or None,
        "charset": "utf8mb4" if charset.lower() == "utf8" else charset,
        "cursorclass_name": "DictCursor",
        "autocommit": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    apply_local_overrides(settings)
    return settings


def apply_local_overrides(settings: Settings) -> None:
    """Read optional python_backend/application-local.yml for local overrides."""

    path = settings.repo_root / "python_backend" / "application-local.yml"
    if not path.exists():
        return
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return
    yshopping = data.get("yshopping") or {}
    llm = yshopping.get("llm") or {}
    embedding = yshopping.get("embedding") or {}

    if not os.getenv("OPENAI_BASE_URL") and not os.getenv("YSHOPPING_LLM_BASE_URL"):
        settings.llm_base_url = resolve_placeholder(llm.get("base-url"), settings.llm_base_url)
    if not os.getenv("OPENAI_MODEL") and not os.getenv("YSHOPPING_LLM_MODEL"):
        settings.llm_model = resolve_placeholder(llm.get("model"), settings.llm_model)
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("YSHOPPING_LLM_API_KEY"):
        settings.llm_api_key = resolve_placeholder(llm.get("api-key"), settings.llm_api_key)

    if not os.getenv("YSHOPPING_EMBEDDING_BASE_URL"):
        settings.embedding_base_url = resolve_placeholder(embedding.get("base-url"), settings.embedding_base_url)
    if not os.getenv("YSHOPPING_EMBEDDING_MODEL"):
        settings.embedding_model = resolve_placeholder(embedding.get("model"), settings.embedding_model)
    if not os.getenv("YSHOPPING_EMBEDDING_API_KEY"):
        settings.embedding_api_key = resolve_placeholder(embedding.get("api-key"), settings.embedding_api_key)


def resolve_placeholder(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    match = re.fullmatch(r"\$\{([^:}]+):?([^}]*)\}", text)
    if not match:
        return text
    env_name, fallback = match.groups()
    return os.getenv(env_name, fallback or default)
