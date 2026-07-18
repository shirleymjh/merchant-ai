# yshopping Merchant AI Python Backend

这是 yshopping Merchant AI 的 Python 后端入口，采用 FastAPI + Deep Agents + LangChain/LangGraph 的通用 ReAct runtime，并以单 Grounded Core、受治理 Query Contract 和 FileSystem-as-Context 为在线查数主线。

要求 Python `>=3.11,<4.0`。Deep Agents、LangChain/Core、LangGraph 和 checkpoint 包必须按 `pyproject.toml` 中的兼容组一起升级，不能复用 Python 3.9 + LangChain 0.3 的旧虚拟环境。

## 结构

- `app/main.py`：FastAPI Gateway，保持 `/api/chat`、`/api/daily-report`、`/api/topics/*` 等路径兼容。
- `merchant_ai/services/grounded_deep_agent_runtime.py`：唯一在线 Core ReAct、Goal Contract、批量查询工具、运行预算、Skill 后置隔离和会话 checkpoint。
- `merchant_ai/services/grounded_runtime_kernel.py`：查询 Contract 的事务状态、安全门、隔离 branch、verified artifact ledger 和回答证据组合；它不是 workflow scheduler。
- `merchant_ai/services/grounded_query_contract.py`：语义绑定 Contract 与单表确定性 SQL 编译。
- `merchant_ai/services/grounded_query_executor.py`：SQL AST/权限校验、租户与 entity-set 注入、Doris 执行。
- `merchant_ai/services/grounded_goal_contract.py`：原问题目标账本、依赖闭包和最终 coverage gate。
- `merchant_ai/services/grounded_runtime_budget.py`：90 秒、LLM、工具和 Doris 共享预算及阶段遥测。
- `merchant_ai/graph/workflow.py` 与旧 `deep_agent_runtime.py`：仅保留历史/离线兼容代码，不再拥有在线查询 authority。
- `langgraph.json`：LangGraph Studio / server 可识别的 graph 配置。

## FileSystem-as-Context

运行时按 `L0 Topic manifest -> L1 表详情 -> L2 指标/字段/schema -> L3 relationship/规则` 渐进披露。Core 自行选择 `ls/read_file/grep`；只有成功读取的 Topic-scoped 精确证据可以被 Grounded Query Contract 引用。旧 Planner 文件工具循环不参与在线查询 authority。

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

## 历史分布式 Worker（非当前在线查数主链路）

仓库仍保留旧 NodeWorker、SkillWorker、文档分析和受控 Python Batch 的 Durable Worker 能力，但当前 `GroundedDeepAgentRuntime` 的在线查数不向这些 worker 派发查询任务。以下配置只适用于仍显式使用旧 worker 的离线/兼容场景：

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

## 当前 Grounded 查询行为

- 单 Core 根据 observation 动态决定语义补读、Contract 数量、串并行、SQL repair 和最终化；没有 action catalog 或固定 `retrieve -> plan -> execute` workflow。
- Original Question Goal Contract 先锁定“必须答什么”，每个查询声明自己覆盖的 `goalIds`。
- Grounded Query Contract 是唯一语义规划 authority。简单单表形状确定性编译；复杂 SQL 由 Core 完整提交并接受 AST、lineage、relationship 和权限校验。
- 独立查询可以放入 batch。每个 branch 拥有独立 generation，线程池只并发执行/验证数据库查询，只合并 verified artifacts。
- `upstreamEntityBindings` 引用的 entity chain 必须等上游 verified entity set 发布后串行执行。
- 最终回答和后置 Skill 都必须先通过 verified portfolio 与原问题 Goal coverage gate；partial evidence 不能伪装成完整答案。
- Debug trace 暴露运行总预算、按名称调用次数与各阶段耗时，便于定位慢在 Core、工具、Doris 还是 Evidence。

当前仓库以后端 Python-only 为准。
