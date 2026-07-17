# yshopping Merchant AI Python Backend

这是 yshopping Merchant AI 的 Python 后端入口，采用 FastAPI + Deep Agents + LangChain + LangGraph，并以 Diana 的 Lead Agent、受治理 QueryGraph 和 FileSystem-as-Context 为迁移主线。

要求 Python `>=3.11,<4.0`。Deep Agents、LangChain/Core、LangGraph 和 checkpoint 包必须按 `pyproject.toml` 中的兼容组一起升级，不能复用 Python 3.9 + LangChain 0.3 的旧虚拟环境。

## 结构

- `app/main.py`：FastAPI Gateway，保持 `/api/chat`、`/api/daily-report`、`/api/topics/*` 等路径兼容。
- `merchant_ai/services/deep_agent_runtime.py`：唯一 Core ReAct、原生文件工具、短期上下文、摘要/offload、subagent 隔离及会话 checkpoint。
- `merchant_ai/graph/workflow.py`：Diana domain kernel，提供受治理的 Topic、QueryGraph、SQL、证据、记忆和回答动作。
- `merchant_ai/services/`：路由、检索、资产包压缩、QueryGraph 规划/校验、NodeWorker SQL 执行、回答组装。
- `langgraph.json`：LangGraph Studio / server 可识别的 graph 配置。

## FileSystem-as-Context

运行时按 `L0 表清单 -> L1 表详情 -> L2 指标/字段/schema -> L3 关系/规则` 渐进披露。Core 自行选择 `ls/read_file/grep`；Planner 只消费 Core 成功读取后形成的 Topic-scoped 可信证据账本。默认 `YSHOPPING_AGENT_PLANNER_TOOL_ROUNDS=0`、`YSHOPPING_PLANNER_FILESYSTEM_CONTEXT_MODE=off`，旧 Planner 文件工具循环仅保留为显式兼容路径。Domain 产生的 QueryGraph、SQL/result 和证据文件从原存储只读挂载为 `/artifacts`，Core 可按需翻阅而无需重新注入整份结果。

DeepAgent 负责短期 message state、checkpoint、自动摘要、state filesystem 与大结果 offload。个人 memory 由 domain store 做租户隔离、按题选择、TTL、冲突和反馈后自动沉淀，使用 `write/reject/quarantine` 自动门禁而非人工审核；共享指标口径/规则/术语则进入独立 knowledge `confirm/review/publish` 生命周期，只有显式 platform/shared scope 才能发布。DeepAgent 的 `AGENTS.md` MemoryMiddleware 无法保持这一区分，因此不启用，并禁止模型直接写入 `/memory`。

## 启动

```bash
cd python_backend
python3.11 -m venv .venv-deepagent
source .venv-deepagent/bin/activate
pip install -e .
python -m merchant_ai.runtime_compat
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

前端仍然访问同样的 `/api/*` 路径。若前端 Vite dev server 代理到 `8088`，无需改前端代码。

## 关键环境变量

关键变量：

```bash
export YSHOPPING_LLM_BASE_URL="https://api.openai.com/v1"
export YSHOPPING_LLM_MODEL="gpt-5.2"
export YSHOPPING_LLM_API_KEY="..."
export YSHOPPING_DORIS_JDBC_URL="jdbc:mysql://127.0.0.1:9030/yshopping?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai"
export YSHOPPING_DORIS_USERNAME="root"
export YSHOPPING_DORIS_PASSWORD=""
export YSHOPPING_ANSWER_JDBC_URL="$YSHOPPING_DORIS_JDBC_URL"
export YSHOPPING_ANSWER_USERNAME="root"
export YSHOPPING_ANSWER_PASSWORD=""
```

也兼容 `OPENAI_BASE_URL`、`OPENAI_MODEL`、`OPENAI_API_KEY`。

轻量前置路由可单独使用小模型 API，不配置时会回退到 `YSHOPPING_LLM_FAST_MODEL`，再回退到主模型：

```bash
export YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_ENABLED=true
export YSHOPPING_PREFLIGHT_LLM_BASE_URL="https://api.openai.com/v1"
export YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_MODEL="gpt-4o-mini"
export YSHOPPING_PREFLIGHT_LLM_API_KEY="..."
export YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_TIMEOUT_SECONDS=3
export YSHOPPING_MEMORY_QUERY_UNDERSTANDING_ENABLED=true
export YSHOPPING_MEMORY_QUERY_UNDERSTANDING_TIMEOUT_SECONDS=2
```

如果只是同一个 API 下切小模型，也可以只配：

```bash
export YSHOPPING_LLM_FAST_MODEL="gpt-4o-mini"
```

长期记忆召回的 query understanding 复用这组轻量路由小模型配置，只生成 `queryVariants`、`expandedTerms`、候选指标和候选意图；结果会进入 TTL/Redis 缓存，并在单次请求内写入 state，避免重复调用。

可选 Redis 配置：

```bash
export YSHOPPING_REDIS_ENABLED=true
export YSHOPPING_REDIS_URL="redis://127.0.0.1:6379/0"
export YSHOPPING_REDIS_NAMESPACE="yshopping_merchant_ai"
export YSHOPPING_REDIS_CACHE_ENABLED=true
export YSHOPPING_REDIS_RATE_LIMIT_ENABLED=true
```

Redis 默认关闭。开启后，召回、语义资产包、Doris SELECT、LLM 响应、工具语义缓存会优先写 Redis；ToolRuntime 的工具缓存和限流状态也会跨实例共享。Redis 不可用时会回退到进程内内存缓存，避免本地开发环境直接启动失败。

## 分布式 Sub-Agent Worker

NodeWorker、SkillWorker、假设复核、文档分析和受控 Python Batch 可以提交到独立 Durable Worker。生产环境建议使用 Redis 或 Postgres 状态后端：

```bash
export YSHOPPING_DISTRIBUTED_SUBAGENTS_ENABLED=true
export YSHOPPING_RUNTIME_STATE_BACKEND=redis
export YSHOPPING_REDIS_URL=redis://redis:6379/0
export YSHOPPING_DISTRIBUTED_WORKER_EXECUTION_BACKEND=process
python -m merchant_ai.worker
```

Worker 会跨 Run 拉取队列、原子 claim、续租 heartbeat，并在租约过期后重试。每个任务在独立子进程执行；取消或超时会终止子进程，不会继续占用 LLM/Doris。可用 `--kinds query_node,analysis_skill` 限制 Worker 类型，或用 `--once` 执行单个任务。

容器部署：

```bash
docker build -f Dockerfile.worker -t merchant-ai-worker .
docker run --rm --env-file .env merchant-ai-worker
```

跨实例结果默认写共享文件系统；也可切换到 S3/MinIO：

```bash
pip install -e '.[distributed]'
export YSHOPPING_DISTRIBUTED_ARTIFACT_BACKEND=s3
export YSHOPPING_DISTRIBUTED_ARTIFACT_S3_BUCKET=merchant-ai-artifacts
export YSHOPPING_DISTRIBUTED_ARTIFACT_S3_PREFIX=merchant-ai/prod
export YSHOPPING_DISTRIBUTED_ARTIFACT_S3_ENDPOINT=http://minio:9000  # AWS S3 可不设置
```

任务请求和结果只在状态表保存 Artifact URI，完整内容保存在 Artifact Store。API 实例和 Worker 因而无需共享进程内对象。

## V2 QueryGraph 行为

Python 版按附件里的 V2 思路实现：

- DeepAgent Core 根据安全 action catalog 自主选择动作，不固定写死 `retrieve -> plan -> execute`。
- `retrieve_knowledge`、文件补读和 `plan_graph` 可由 Core 根据 Planner/validator/execution observation 在有界预算内重新触发。
- `compact_assets` 从 `python_backend/resources/runtime/topics` 生成 `PlanningAssetPack`。
- `plan_query_graph` 是一次结构化规划编译，不拥有文件工具；证据不足会将 typed knowledge request 交还 Core。
- `validate_query_graph` 做表、字段、relationship 硬门禁。
- `execute_query_graph` 按 NodeWorker 执行 node，SQL 优先 LLM 生成并经过只读 SQL 校验。
- 新鲜度、零行和 SQL 修复结果会回到 Core observation；缺口可触发换表、补读和重规划。
- `answer_analysis` 只基于 evidence 回答，partial evidence 会保留 gap。

当前仓库以后端 Python-only 为准。
