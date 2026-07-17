from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


ENV_FILE = Path(__file__).resolve().parents[2] / "python_backend" / ".env"


@dataclass(frozen=True)
class SecuritySettings:
    ops_token: str
    cors_allow_origins: list[str]
    cors_allow_credentials: bool
    identity_auth_required: bool
    identity_jwt_secret_configured: bool


@dataclass(frozen=True)
class LlmSettings:
    base_url: str
    model: str
    api_key: str
    request_timeout_seconds: int
    lead_timeout_seconds: int
    planner_timeout_seconds: int
    answer_timeout_seconds: int
    analysis_timeout_seconds: int
    max_tokens: int


@dataclass(frozen=True)
class EsSettings:
    enabled: bool
    base_url: str
    index: str
    vector_enabled: bool
    vector_field: str
    hybrid_top_k: int


@dataclass(frozen=True)
class DorisSettings:
    jdbc_url: str
    username: str
    password: str
    read_timeout_seconds: int


@dataclass(frozen=True)
class MemorySettings:
    backend: str
    es_index: str
    vector_enabled: bool
    cache_ttl_seconds: int


@dataclass(frozen=True)
class AgentSettings:
    mode: str
    max_concurrent_sub_agents: int
    main_rounds: int
    trace_replay_enabled: bool
    checkpointer_backend: str


@dataclass(frozen=True)
class RuntimeSettings:
    workspace_path: Path
    context_window_tokens: int
    cache_enabled: bool
    redis_enabled: bool
    state_backend: str


class Settings(BaseSettings):
    """Runtime configuration for the Python-only merchant AI backend."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore", populate_by_name=True)

    server_port: int = Field(8088, validation_alias="SERVER_PORT")
    company_name: str = Field("yshopping", validation_alias="YSHOPPING_COMPANY_NAME")
    business_timezone: str = Field("Asia/Shanghai", validation_alias="YSHOPPING_BUSINESS_TIMEZONE")
    calendar_time_semantics_enabled: bool = Field(True, validation_alias="YSHOPPING_CALENDAR_TIME_SEMANTICS_ENABLED")
    merchant_id: str = Field("100", validation_alias="YSHOPPING_MERCHANT_ID")
    allowed_merchant_ids: str = Field("", validation_alias="YSHOPPING_ALLOWED_MERCHANT_IDS")
    ops_token: str = Field("", validation_alias="YSHOPPING_OPS_TOKEN")
    identity_auth_required: bool = Field(False, validation_alias="YSHOPPING_IDENTITY_AUTH_REQUIRED")
    identity_jwt_secret: str = Field("", validation_alias="YSHOPPING_IDENTITY_JWT_SECRET")
    identity_jwt_issuer: str = Field("", validation_alias="YSHOPPING_IDENTITY_JWT_ISSUER")
    identity_jwt_audience: str = Field("", validation_alias="YSHOPPING_IDENTITY_JWT_AUDIENCE")
    cors_allow_origins: str = Field(
        "http://localhost:5173,http://127.0.0.1:5173",
        validation_alias="YSHOPPING_CORS_ALLOW_ORIGINS",
    )
    cors_allow_credentials: bool = Field(False, validation_alias="YSHOPPING_CORS_ALLOW_CREDENTIALS")

    llm_base_url: str = Field("https://api.openai.com/v1", validation_alias="YSHOPPING_LLM_BASE_URL")
    llm_model: str = Field("gpt-5.2", validation_alias="YSHOPPING_LLM_MODEL")
    llm_api_key: str = Field("", validation_alias="YSHOPPING_LLM_API_KEY")
    llm_request_timeout_seconds: int = Field(12, validation_alias="YSHOPPING_LLM_REQUEST_TIMEOUT_SECONDS")
    llm_lead_timeout_seconds: int = Field(12, validation_alias="YSHOPPING_LLM_LEAD_TIMEOUT_SECONDS")
    llm_planner_timeout_seconds: int = Field(20, validation_alias="YSHOPPING_LLM_PLANNER_TIMEOUT_SECONDS")
    llm_answer_timeout_seconds: int = Field(10, validation_alias="YSHOPPING_LLM_ANSWER_TIMEOUT_SECONDS")
    llm_analysis_timeout_seconds: int = Field(12, validation_alias="YSHOPPING_LLM_ANALYSIS_TIMEOUT_SECONDS")
    llm_circuit_threshold: int = Field(3, validation_alias="YSHOPPING_LLM_CIRCUIT_THRESHOLD")
    llm_circuit_cooldown_seconds: int = Field(30, validation_alias="YSHOPPING_LLM_CIRCUIT_COOLDOWN_SECONDS")
    llm_max_tokens: int = Field(2048, validation_alias="YSHOPPING_LLM_MAX_TOKENS")
    llm_strong_model: str = Field("", validation_alias="YSHOPPING_LLM_STRONG_MODEL")
    llm_balanced_model: str = Field("", validation_alias="YSHOPPING_LLM_BALANCED_MODEL")
    llm_fast_model: str = Field("", validation_alias="YSHOPPING_LLM_FAST_MODEL")
    preflight_semantic_route_enabled: bool = Field(True, validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_ENABLED")
    preflight_llm_base_url: str = Field("", validation_alias="YSHOPPING_PREFLIGHT_LLM_BASE_URL")
    preflight_llm_api_key: str = Field("", validation_alias="YSHOPPING_PREFLIGHT_LLM_API_KEY")
    preflight_semantic_route_model: str = Field("", validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_MODEL")
    preflight_semantic_route_timeout_seconds: int = Field(3, validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_TIMEOUT_SECONDS")
    preflight_semantic_route_max_timeout_seconds: int = Field(5, validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_MAX_TIMEOUT_SECONDS")
    preflight_semantic_route_min_confidence: float = Field(0.62, validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_MIN_CONFIDENCE")
    preflight_semantic_route_high_confidence: float = Field(0.86, validation_alias="YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_HIGH_CONFIDENCE")
    answer_skill_match_mode: str = Field("always", validation_alias="YSHOPPING_ANSWER_SKILL_MATCH_MODE")
    always_apply_rule_budget: int = Field(20, validation_alias="YSHOPPING_ALWAYS_APPLY_RULE_BUDGET")
    skill_confirmation_required: bool = Field(False, validation_alias="YSHOPPING_SKILL_CONFIRMATION_REQUIRED")
    skill_reuse_suggestion_enabled: bool = Field(True, validation_alias="YSHOPPING_SKILL_REUSE_SUGGESTION_ENABLED")
    skill_worker_enabled: bool = Field(True, validation_alias="YSHOPPING_SKILL_WORKER_ENABLED")
    skill_worker_timeout_seconds: int = Field(10, validation_alias="YSHOPPING_SKILL_WORKER_TIMEOUT_SECONDS")
    sandbox_backend: str = Field("local", validation_alias="YSHOPPING_SANDBOX_BACKEND")
    sandbox_container_runtime: str = Field("docker", validation_alias="YSHOPPING_SANDBOX_CONTAINER_RUNTIME")
    sandbox_container_image: str = Field("python:3.11-slim-bookworm", validation_alias="YSHOPPING_SANDBOX_CONTAINER_IMAGE")
    sandbox_container_memory: str = Field("512m", validation_alias="YSHOPPING_SANDBOX_CONTAINER_MEMORY")
    sandbox_container_cpus: float = Field(1.0, validation_alias="YSHOPPING_SANDBOX_CONTAINER_CPUS")
    skill_worker_parallel_enabled: bool = Field(True, validation_alias="YSHOPPING_SKILL_WORKER_PARALLEL_ENABLED")
    max_concurrent_skill_workers: int = Field(2, validation_alias="YSHOPPING_MAX_CONCURRENT_SKILL_WORKERS")
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
    es_vector_enabled: bool = Field(True, validation_alias="YSHOPPING_ES_VECTOR_ENABLED")
    es_vector_field: str = Field("content_vector", validation_alias="YSHOPPING_ES_VECTOR_FIELD")
    es_text_top_k: int = Field(12, validation_alias="YSHOPPING_ES_TEXT_TOP_K")
    es_broad_text_top_k: int = Field(4, validation_alias="YSHOPPING_ES_BROAD_TEXT_TOP_K")
    es_vector_top_k: int = Field(12, validation_alias="YSHOPPING_ES_VECTOR_TOP_K")
    es_broad_vector_top_k: int = Field(4, validation_alias="YSHOPPING_ES_BROAD_VECTOR_TOP_K")
    es_vector_num_candidates: int = Field(80, validation_alias="YSHOPPING_ES_VECTOR_NUM_CANDIDATES")
    es_rrf_k: int = Field(60, validation_alias="YSHOPPING_ES_RRF_K")
    es_rrf_score_scale: float = Field(1000.0, validation_alias="YSHOPPING_ES_RRF_SCORE_SCALE")
    es_hybrid_top_k: int = Field(24, validation_alias="YSHOPPING_ES_HYBRID_TOP_K")
    es_retrieval_profiles_json: str = Field("", validation_alias="YSHOPPING_ES_RETRIEVAL_PROFILES_JSON")
    rule_chunk_target_chars: int = Field(1600, validation_alias="YSHOPPING_RULE_CHUNK_TARGET_CHARS")
    rule_chunk_max_chars: int = Field(2400, validation_alias="YSHOPPING_RULE_CHUNK_MAX_CHARS")
    rule_chunk_overlap_chars: int = Field(160, validation_alias="YSHOPPING_RULE_CHUNK_OVERLAP_CHARS")

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

    memory_backend: str = Field("es", validation_alias="YSHOPPING_MEMORY_BACKEND")
    memory_mysql_jdbc_url: str = Field("", validation_alias="YSHOPPING_MEMORY_MYSQL_JDBC_URL")
    memory_mysql_username: str = Field("", validation_alias="YSHOPPING_MEMORY_MYSQL_USERNAME")
    memory_mysql_password: str = Field("", validation_alias="YSHOPPING_MEMORY_MYSQL_PASSWORD")
    memory_redis_enabled: bool = Field(False, validation_alias="YSHOPPING_MEMORY_REDIS_ENABLED")
    memory_vector_enabled: bool = Field(False, validation_alias="YSHOPPING_MEMORY_VECTOR_ENABLED")
    memory_cache_ttl_seconds: int = Field(600, validation_alias="YSHOPPING_MEMORY_CACHE_TTL_SECONDS")
    memory_index_async: bool = Field(True, validation_alias="YSHOPPING_MEMORY_INDEX_ASYNC")
    memory_es_index: str = Field("merchant_memory", validation_alias="YSHOPPING_MEMORY_ES_INDEX")
    memory_vector_index: str = Field("merchant_memory", validation_alias="YSHOPPING_MEMORY_VECTOR_INDEX")
    memory_query_understanding_enabled: bool = Field(True, validation_alias="YSHOPPING_MEMORY_QUERY_UNDERSTANDING_ENABLED")
    memory_query_understanding_timeout_seconds: int = Field(4, validation_alias="YSHOPPING_MEMORY_QUERY_UNDERSTANDING_TIMEOUT_SECONDS")
    memory_query_understanding_max_timeout_seconds: int = Field(5, validation_alias="YSHOPPING_MEMORY_QUERY_UNDERSTANDING_MAX_TIMEOUT_SECONDS")
    memory_curator_enabled: bool = Field(True, validation_alias="YSHOPPING_MEMORY_CURATOR_ENABLED")
    memory_curator_timeout_seconds: int = Field(8, validation_alias="YSHOPPING_MEMORY_CURATOR_TIMEOUT_SECONDS")
    memory_curator_min_confidence: float = Field(0.72, validation_alias="YSHOPPING_MEMORY_CURATOR_MIN_CONFIDENCE")
    memory_curator_max_candidates: int = Field(3, validation_alias="YSHOPPING_MEMORY_CURATOR_MAX_CANDIDATES")
    knowledge_conflict_enabled: bool = Field(True, validation_alias="YSHOPPING_KNOWLEDGE_CONFLICT_ENABLED")
    knowledge_conflict_timeout_seconds: int = Field(6, validation_alias="YSHOPPING_KNOWLEDGE_CONFLICT_TIMEOUT_SECONDS")
    knowledge_conflict_top_k: int = Field(5, validation_alias="YSHOPPING_KNOWLEDGE_CONFLICT_TOP_K")
    knowledge_conflict_min_similarity: float = Field(0.18, validation_alias="YSHOPPING_KNOWLEDGE_CONFLICT_MIN_SIMILARITY")

    harness_workspace_path: str = Field("", validation_alias="YSHOPPING_HARNESS_WORKSPACE")
    thread_context_summary_ttl_seconds: int = Field(
        2592000,
        validation_alias="YSHOPPING_THREAD_CONTEXT_SUMMARY_TTL_SECONDS",
    )
    context_window_tokens: int = Field(16000, validation_alias="YSHOPPING_HARNESS_CONTEXT_WINDOW_TOKENS")
    tool_result_preview_rows: int = Field(20, validation_alias="YSHOPPING_HARNESS_TOOL_RESULT_PREVIEW_ROWS")
    max_sub_agent_tasks: int = Field(3, validation_alias="YSHOPPING_HARNESS_MAX_SUB_AGENT_TASKS")
    max_concurrent_sub_agents: int = Field(3, validation_alias="YSHOPPING_HARNESS_MAX_CONCURRENT_SUB_AGENTS")
    hypothesis_query_exploration_enabled: bool = Field(True, validation_alias="YSHOPPING_HYPOTHESIS_QUERY_EXPLORATION_ENABLED")
    hypothesis_max_candidates: int = Field(2, validation_alias="YSHOPPING_HYPOTHESIS_MAX_CANDIDATES")
    hypothesis_max_rounds: int = Field(2, validation_alias="YSHOPPING_HYPOTHESIS_MAX_ROUNDS")
    hypothesis_max_parallel_queries: int = Field(3, validation_alias="YSHOPPING_HYPOTHESIS_MAX_PARALLEL_QUERIES")
    hypothesis_min_survivor_score: int = Field(45, validation_alias="YSHOPPING_HYPOTHESIS_MIN_SURVIVOR_SCORE")
    hypothesis_max_survivors: int = Field(2, validation_alias="YSHOPPING_HYPOTHESIS_MAX_SURVIVORS")
    hypothesis_second_round_min_information_gain: float = Field(0.35, validation_alias="YSHOPPING_HYPOTHESIS_SECOND_ROUND_MIN_INFORMATION_GAIN")
    hypothesis_answer_reserve_seconds: int = Field(15, validation_alias="YSHOPPING_HYPOTHESIS_ANSWER_RESERVE_SECONDS")
    max_sub_agent_rounds: int = Field(6, validation_alias="YSHOPPING_HARNESS_MAX_SUB_AGENT_ROUNDS")
    tool_max_concurrency: int = Field(4, validation_alias="YSHOPPING_TOOL_MAX_CONCURRENCY")
    tool_failure_repeat_threshold: int = Field(2, validation_alias="YSHOPPING_TOOL_FAILURE_REPEAT_THRESHOLD")
    tool_circuit_threshold: int = Field(5, validation_alias="YSHOPPING_TOOL_CIRCUIT_THRESHOLD")
    tool_circuit_cooldown_seconds: int = Field(60, validation_alias="YSHOPPING_TOOL_CIRCUIT_COOLDOWN_SECONDS")
    tool_rate_limit_enabled: bool = Field(True, validation_alias="YSHOPPING_TOOL_RATE_LIMIT_ENABLED")
    tool_default_qps: int = Field(5, validation_alias="YSHOPPING_TOOL_DEFAULT_QPS")
    tool_heartbeat_interval_seconds: float = Field(5.0, validation_alias="YSHOPPING_TOOL_HEARTBEAT_INTERVAL_SECONDS")
    semantic_cache_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_SEMANTIC_CACHE_TTL_SECONDS")
    llm_targets: str = Field("", validation_alias="YSHOPPING_LLM_TARGETS")
    es_targets: str = Field("", validation_alias="YSHOPPING_ES_TARGETS")
    doris_targets: str = Field("", validation_alias="YSHOPPING_DORIS_TARGETS")
    alert_webhook_url: str = Field("", validation_alias="YSHOPPING_ALERT_WEBHOOK_URL")
    alert_p95_threshold_ms: int = Field(30000, validation_alias="YSHOPPING_ALERT_P95_THRESHOLD_MS")
    agent_v2_enabled: bool = Field(True, validation_alias="YSHOPPING_HARNESS_AGENT_V2_ENABLED")
    agent_mode: str = Field("deepagent", validation_alias="YSHOPPING_AGENT_MODE")
    agent_node_timeout_seconds: int = Field(45, validation_alias="YSHOPPING_AGENT_NODE_TIMEOUT_SECONDS")
    agent_node_poll_interval_seconds: float = Field(5.0, validation_alias="YSHOPPING_AGENT_NODE_POLL_INTERVAL_SECONDS")
    agent_node_timeout_grace_seconds: int = Field(60, validation_alias="YSHOPPING_AGENT_NODE_TIMEOUT_GRACE_SECONDS")
    agent_sql_repair_rounds: int = Field(2, validation_alias="YSHOPPING_AGENT_SQL_REPAIR_ROUNDS")
    agent_max_entity_values: int = Field(200, validation_alias="YSHOPPING_AGENT_MAX_ENTITY_VALUES")
    agent_doris_split_query_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_DORIS_SPLIT_QUERY_ENABLED")
    agent_doris_split_chunk_days: int = Field(7, validation_alias="YSHOPPING_AGENT_DORIS_SPLIT_CHUNK_DAYS")
    agent_doris_split_max_chunks: int = Field(6, validation_alias="YSHOPPING_AGENT_DORIS_SPLIT_MAX_CHUNKS")
    agent_doris_split_max_concurrency: int = Field(3, validation_alias="YSHOPPING_AGENT_DORIS_SPLIT_MAX_CONCURRENCY")
    agent_partition_date_anchor_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_PARTITION_DATE_ANCHOR_ENABLED")
    agent_trace_replay_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_TRACE_REPLAY_ENABLED")
    agent_compact_success_artifacts_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_COMPACT_SUCCESS_ARTIFACTS_ENABLED")
    agent_run_retention_days: int = Field(14, validation_alias="YSHOPPING_AGENT_RUN_RETENTION_DAYS")
    agent_completed_checkpoint_limit: int = Field(30, validation_alias="YSHOPPING_AGENT_COMPLETED_CHECKPOINT_LIMIT")
    attachment_retention_days: int = Field(7, validation_alias="YSHOPPING_ATTACHMENT_RETENTION_DAYS")
    attachment_max_bytes: int = Field(20 * 1024 * 1024, validation_alias="YSHOPPING_ATTACHMENT_MAX_BYTES")
    attachment_preview_max_chars: int = Field(12000, validation_alias="YSHOPPING_ATTACHMENT_PREVIEW_MAX_CHARS")
    agent_checkpointer_backend: str = Field("sqlite", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_BACKEND")
    agent_checkpointer_sqlite_path: str = Field("", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_SQLITE_PATH")
    agent_checkpointer_postgres_uri: str = Field("", validation_alias="YSHOPPING_AGENT_CHECKPOINTER_POSTGRES_URI")
    agent_main_rounds: int = Field(16, validation_alias="YSHOPPING_AGENT_MAIN_ROUNDS")
    agent_retrieve_rounds: int = Field(3, validation_alias="YSHOPPING_AGENT_RETRIEVE_ROUNDS")
    agent_plan_rounds: int = Field(1, validation_alias="YSHOPPING_AGENT_PLAN_ROUNDS")
    agent_graph_repair_rounds: int = Field(2, validation_alias="YSHOPPING_AGENT_GRAPH_REPAIR_ROUNDS")
    agent_lead_action_retries: int = Field(1, validation_alias="YSHOPPING_AGENT_LEAD_ACTION_RETRIES")
    agent_planner_transient_retries: int = Field(1, validation_alias="YSHOPPING_AGENT_PLANNER_TRANSIENT_RETRIES")
    # Compatibility budget for the pre-DeepAgent hidden Planner file-tool
    # loop. Diana's default runtime keeps this disabled: the outer Core owns
    # ls/grep/read_file and Planner receives only the trusted read ledger.
    agent_planner_tool_rounds: int = Field(0, validation_alias="YSHOPPING_AGENT_PLANNER_TOOL_ROUNDS")
    agent_planner_invalid_output_retries: int = Field(
        1,
        validation_alias="YSHOPPING_AGENT_PLANNER_INVALID_OUTPUT_RETRIES",
    )
    planner_semantic_contract_compile_enabled: bool = Field(
        True,
        validation_alias="YSHOPPING_PLANNER_SEMANTIC_CONTRACT_COMPILE_ENABLED",
    )
    agent_deferred_tool_schema_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_DEFERRED_TOOL_SCHEMA_ENABLED")
    planner_filesystem_context_mode: str = Field("off", validation_alias="YSHOPPING_PLANNER_FILESYSTEM_CONTEXT_MODE")
    legacy_question_planner_enabled: bool = Field(False, validation_alias="YSHOPPING_LEGACY_QUESTION_PLANNER_ENABLED")
    lead_action_llm_mode: str = Field("adaptive", validation_alias="YSHOPPING_LEAD_ACTION_LLM_MODE")
    lead_agent_autonomous_enabled: bool = Field(True, validation_alias="YSHOPPING_LEAD_AGENT_AUTONOMOUS_ENABLED")
    agent_planner_seed_table_limit: int = Field(4, validation_alias="YSHOPPING_AGENT_PLANNER_SEED_TABLE_LIMIT")
    agent_planner_seed_metric_limit: int = Field(14, validation_alias="YSHOPPING_AGENT_PLANNER_SEED_METRIC_LIMIT")
    agent_asset_field_entry_limit: int = Field(240, validation_alias="YSHOPPING_AGENT_ASSET_FIELD_ENTRY_LIMIT")
    agent_planner_prompt_budget_chars: int = Field(30000, validation_alias="YSHOPPING_AGENT_PLANNER_PROMPT_BUDGET_CHARS")
    agent_node_file_tool_rounds: int = Field(1, validation_alias="YSHOPPING_AGENT_NODE_FILE_TOOL_ROUNDS")
    answer_file_tool_rounds: int = Field(1, validation_alias="YSHOPPING_ANSWER_FILE_TOOL_ROUNDS")
    route_llm_mode: str = Field("low_confidence", validation_alias="YSHOPPING_ROUTE_LLM_MODE")
    route_force_clarification_enabled: bool = Field(True, validation_alias="YSHOPPING_ROUTE_FORCE_CLARIFICATION_ENABLED")
    merchant_clarification_enabled: bool = Field(True, validation_alias="YSHOPPING_MERCHANT_CLARIFICATION_ENABLED")
    route_topic_min_confidence: float = Field(0.52, validation_alias="YSHOPPING_ROUTE_TOPIC_MIN_CONFIDENCE")
    route_topic_high_confidence: float = Field(0.75, validation_alias="YSHOPPING_ROUTE_TOPIC_HIGH_CONFIDENCE")
    route_topic_max_candidates: int = Field(4, validation_alias="YSHOPPING_ROUTE_TOPIC_MAX_CANDIDATES")
    route_mixed_rule_data_min_confidence: float = Field(0.75, validation_alias="YSHOPPING_ROUTE_MIXED_RULE_DATA_MIN_CONFIDENCE")
    context_file_inline_max_chars: int = Field(12000, validation_alias="YSHOPPING_AGENT_CONTEXT_FILE_INLINE_MAX_CHARS")
    context_artifact_inline_max_rows: int = Field(20, validation_alias="YSHOPPING_AGENT_CONTEXT_ARTIFACT_INLINE_MAX_ROWS")
    context_compaction_threshold_ratio: float = Field(0.85, validation_alias="YSHOPPING_CONTEXT_COMPACTION_THRESHOLD_RATIO")
    context_compaction_target_ratio: float = Field(0.4, validation_alias="YSHOPPING_CONTEXT_COMPACTION_TARGET_RATIO")
    context_runtime_budget_chars: int = Field(6000, validation_alias="YSHOPPING_CONTEXT_RUNTIME_BUDGET_CHARS")
    context_planner_budget_chars: int = Field(12000, validation_alias="YSHOPPING_CONTEXT_PLANNER_BUDGET_CHARS")
    context_answer_budget_chars: int = Field(10000, validation_alias="YSHOPPING_CONTEXT_ANSWER_BUDGET_CHARS")
    context_memory_budget_tokens: int = Field(1200, validation_alias="YSHOPPING_CONTEXT_MEMORY_BUDGET_TOKENS")
    context_memory_budget_chars: int = Field(1800, validation_alias="YSHOPPING_CONTEXT_MEMORY_BUDGET_CHARS")
    middleware_loop_guard_threshold: int = Field(3, validation_alias="YSHOPPING_MIDDLEWARE_LOOP_GUARD_THRESHOLD")
    middleware_tool_repeat_warning_threshold: int = Field(3, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_REPEAT_WARNING_THRESHOLD")
    middleware_tool_repeat_hard_stop_threshold: int = Field(5, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_REPEAT_HARD_STOP_THRESHOLD")
    middleware_tool_type_warning_threshold: int = Field(30, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_TYPE_WARNING_THRESHOLD")
    middleware_tool_type_hard_stop_threshold: int = Field(50, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_TYPE_HARD_STOP_THRESHOLD")
    middleware_tool_loop_window_size: int = Field(20, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_LOOP_WINDOW_SIZE")
    middleware_tool_output_budget_chars: int = Field(12000, validation_alias="YSHOPPING_MIDDLEWARE_TOOL_OUTPUT_BUDGET_CHARS")
    harness_middleware_disabled: str = Field("", validation_alias="YSHOPPING_HARNESS_MIDDLEWARE_DISABLED")
    harness_middleware_order: str = Field("", validation_alias="YSHOPPING_HARNESS_MIDDLEWARE_ORDER")
    run_budget_max_duration_seconds: int = Field(90, validation_alias="YSHOPPING_RUN_BUDGET_MAX_DURATION_SECONDS")
    run_budget_fast_duration_seconds: int = Field(25, validation_alias="YSHOPPING_RUN_BUDGET_FAST_DURATION_SECONDS")
    run_budget_max_actions: int = Field(20, validation_alias="YSHOPPING_RUN_BUDGET_MAX_ACTIONS")
    run_budget_max_llm_calls: int = Field(8, validation_alias="YSHOPPING_RUN_BUDGET_MAX_LLM_CALLS")
    run_budget_max_doris_queries: int = Field(12, validation_alias="YSHOPPING_RUN_BUDGET_MAX_DORIS_QUERIES")
    run_budget_max_tool_calls: int = Field(60, validation_alias="YSHOPPING_RUN_BUDGET_MAX_TOOL_CALLS")
    run_budget_max_estimated_tokens: int = Field(60000, validation_alias="YSHOPPING_RUN_BUDGET_MAX_ESTIMATED_TOKENS")
    middleware_max_manifest_entries: int = Field(200, validation_alias="YSHOPPING_MIDDLEWARE_MAX_MANIFEST_ENTRIES")
    tool_result_offload_chars: int = Field(20000, validation_alias="YSHOPPING_TOOL_RESULT_OFFLOAD_CHARS")
    result_csv_download_min_rows: int = Field(50, validation_alias="YSHOPPING_RESULT_CSV_DOWNLOAD_MIN_ROWS")

    cache_enabled: bool = Field(True, validation_alias="YSHOPPING_AGENT_CACHE_ENABLED")
    cache_memory_max_entries: int = Field(512, validation_alias="YSHOPPING_AGENT_CACHE_MEMORY_MAX_ENTRIES")
    cache_doris_select_ttl_seconds: int = Field(60, validation_alias="YSHOPPING_AGENT_CACHE_DORIS_SELECT_TTL_SECONDS")
    cache_recall_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_RECALL_TTL_SECONDS")
    cache_asset_pack_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_ASSET_PACK_TTL_SECONDS")
    cache_llm_ttl_seconds: int = Field(300, validation_alias="YSHOPPING_AGENT_CACHE_LLM_TTL_SECONDS")
    redis_enabled: bool = Field(False, validation_alias="YSHOPPING_REDIS_ENABLED")
    redis_url: str = Field("redis://127.0.0.1:6379/0", validation_alias="YSHOPPING_REDIS_URL")
    redis_namespace: str = Field("yshopping_merchant_ai", validation_alias="YSHOPPING_REDIS_NAMESPACE")
    redis_socket_timeout_seconds: float = Field(1.0, validation_alias="YSHOPPING_REDIS_SOCKET_TIMEOUT_SECONDS")
    redis_cache_enabled: bool = Field(True, validation_alias="YSHOPPING_REDIS_CACHE_ENABLED")
    redis_rate_limit_enabled: bool = Field(True, validation_alias="YSHOPPING_REDIS_RATE_LIMIT_ENABLED")
    runtime_state_backend: str = Field("file", validation_alias="YSHOPPING_RUNTIME_STATE_BACKEND")
    runtime_state_postgres_uri: str = Field("", validation_alias="YSHOPPING_RUNTIME_STATE_POSTGRES_URI")
    distributed_subagents_enabled: bool = Field(False, validation_alias="YSHOPPING_DISTRIBUTED_SUBAGENTS_ENABLED")
    distributed_worker_poll_seconds: float = Field(0.5, validation_alias="YSHOPPING_DISTRIBUTED_WORKER_POLL_SECONDS")
    distributed_worker_lease_seconds: int = Field(120, validation_alias="YSHOPPING_DISTRIBUTED_WORKER_LEASE_SECONDS")
    distributed_worker_result_timeout_seconds: int = Field(180, validation_alias="YSHOPPING_DISTRIBUTED_WORKER_RESULT_TIMEOUT_SECONDS")
    distributed_worker_max_attempts: int = Field(3, validation_alias="YSHOPPING_DISTRIBUTED_WORKER_MAX_ATTEMPTS")
    distributed_worker_execution_backend: str = Field("process", validation_alias="YSHOPPING_DISTRIBUTED_WORKER_EXECUTION_BACKEND")
    distributed_artifact_backend: str = Field("filesystem", validation_alias="YSHOPPING_DISTRIBUTED_ARTIFACT_BACKEND")
    distributed_artifact_s3_bucket: str = Field("", validation_alias="YSHOPPING_DISTRIBUTED_ARTIFACT_S3_BUCKET")
    distributed_artifact_s3_prefix: str = Field("merchant-ai", validation_alias="YSHOPPING_DISTRIBUTED_ARTIFACT_S3_PREFIX")
    distributed_artifact_s3_endpoint: str = Field("", validation_alias="YSHOPPING_DISTRIBUTED_ARTIFACT_S3_ENDPOINT")
    python_executable: str = Field(default_factory=lambda: sys.executable, validation_alias="YSHOPPING_PYTHON_EXECUTABLE")

    rule_knowledge_path: str = Field("", validation_alias="YSHOPPING_RULE_KNOWLEDGE_PATH")
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
    def resolved_rule_knowledge_path(self) -> Path:
        if self.rule_knowledge_path:
            return Path(self.rule_knowledge_path)
        return self.resources_root / "runtime" / "rules"

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

    @property
    def cors_origins(self) -> list[str]:
        values = [item.strip() for item in str(self.cors_allow_origins or "").split(",") if item.strip()]
        return values or ["*"]

    @property
    def allowed_merchants(self) -> set[str]:
        values = {item.strip() for item in str(self.allowed_merchant_ids or "").split(",") if item.strip()}
        return values or {self.merchant_id}

    def merchant_allowed(self, merchant_id: str) -> bool:
        target = str(merchant_id or self.merchant_id).strip()
        allowed = self.allowed_merchants
        return "*" in allowed or target in allowed

    @property
    def security(self) -> SecuritySettings:
        return SecuritySettings(
            ops_token=self.ops_token,
            cors_allow_origins=self.cors_origins,
            cors_allow_credentials=bool(self.cors_allow_credentials),
            identity_auth_required=bool(self.identity_auth_required),
            identity_jwt_secret_configured=bool(self.identity_jwt_secret),
        )

    @property
    def llm(self) -> LlmSettings:
        return LlmSettings(
            base_url=self.openai_base_url,
            model=self.openai_model,
            api_key=self.openai_api_key,
            request_timeout_seconds=self.llm_request_timeout_seconds,
            lead_timeout_seconds=self.llm_lead_timeout_seconds,
            planner_timeout_seconds=self.llm_planner_timeout_seconds,
            answer_timeout_seconds=self.llm_answer_timeout_seconds,
            analysis_timeout_seconds=self.llm_analysis_timeout_seconds,
            max_tokens=self.llm_max_tokens,
        )

    @property
    def es(self) -> EsSettings:
        return EsSettings(
            enabled=bool(self.es_enabled),
            base_url=self.es_base_url,
            index=self.es_index,
            vector_enabled=bool(self.es_vector_enabled),
            vector_field=self.es_vector_field,
            hybrid_top_k=self.es_hybrid_top_k,
        )

    @property
    def doris(self) -> DorisSettings:
        return DorisSettings(
            jdbc_url=self.doris_jdbc_url,
            username=self.doris_username,
            password=self.doris_password,
            read_timeout_seconds=self.doris_read_timeout_seconds,
        )

    @property
    def memory(self) -> MemorySettings:
        return MemorySettings(
            backend=self.memory_backend,
            es_index=self.memory_es_index,
            vector_enabled=bool(self.memory_vector_enabled),
            cache_ttl_seconds=self.memory_cache_ttl_seconds,
        )

    @property
    def agent(self) -> AgentSettings:
        return AgentSettings(
            mode=self.agent_mode,
            max_concurrent_sub_agents=self.max_concurrent_sub_agents,
            main_rounds=self.agent_main_rounds,
            trace_replay_enabled=bool(self.agent_trace_replay_enabled),
            checkpointer_backend=self.agent_checkpointer_backend,
        )

    @property
    def runtime(self) -> RuntimeSettings:
        return RuntimeSettings(
            workspace_path=self.resolved_workspace_path,
            context_window_tokens=self.context_window_tokens,
            cache_enabled=bool(self.cache_enabled),
            redis_enabled=bool(self.redis_enabled),
            state_backend=self.runtime_state_backend,
        )

    def grouped_summary(self) -> Dict[str, Any]:
        return {
            "security": {
                "opsTokenConfigured": bool(self.security.ops_token),
                "corsAllowOrigins": self.security.cors_allow_origins,
                "corsAllowCredentials": self.security.cors_allow_credentials,
                "allowedMerchants": sorted(self.allowed_merchants),
            },
            "llm": {"baseUrl": self.llm.base_url, "model": self.llm.model, "maxTokens": self.llm.max_tokens},
            "es": {"enabled": self.es.enabled, "index": self.es.index, "hybridTopK": self.es.hybrid_top_k},
            "doris": {"jdbcUrl": self.doris.jdbc_url, "readTimeoutSeconds": self.doris.read_timeout_seconds},
            "memory": {"backend": self.memory.backend, "esIndex": self.memory.es_index},
            "agent": {
                "mode": self.agent.mode,
                "mainRounds": self.agent.main_rounds,
                "checkpointerBackend": self.agent.checkpointer_backend,
            },
            "runtime": {
                "workspacePath": str(self.runtime.workspace_path),
                "contextWindowTokens": self.runtime.context_window_tokens,
                "cacheEnabled": self.runtime.cache_enabled,
                "redisEnabled": self.runtime.redis_enabled,
                "stateBackend": self.runtime.state_backend,
                "distributedSubagentsEnabled": bool(self.distributed_subagents_enabled),
                "distributedWorkerExecutionBackend": self.distributed_worker_execution_backend,
                "distributedArtifactBackend": self.distributed_artifact_backend,
            },
        }


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
