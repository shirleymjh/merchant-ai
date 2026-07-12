# yshopping Merchant AI Python Backend

这是 yshopping Merchant AI 的 Python 后端入口，采用 FastAPI + LangChain + LangGraph。

## 结构

- `app/main.py`：FastAPI Gateway，保持 `/api/chat`、`/api/daily-report`、`/api/topics/*` 等路径兼容。
- `merchant_ai/graph/workflow.py`：LangGraph `StateGraph`，把 V2 Main Agent ReAct Loop 映射成 policy node + action nodes。
- `merchant_ai/services/`：路由、检索、资产包压缩、QueryGraph 规划/校验、NodeWorker SQL 执行、回答组装。
- `langgraph.json`：LangGraph Studio / server 可识别的 graph 配置。

## 启动

```bash
cd python_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
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

- Main Agent 使用 state-driven policy，不固定写死 `retrieve -> plan -> execute`。
- `retrieve_knowledge` 可被 planner/validator gap 反复触发。
- `compact_assets` 从 `python_backend/resources/runtime/topics` 生成 `PlanningAssetPack`。
- `plan_query_graph` 优先由 LLM 生成 QueryGraph；未配置 LLM 时使用安全降级规划。
- `validate_query_graph` 做表、字段、relationship 硬门禁。
- `execute_query_graph` 按 NodeWorker 执行 node，SQL 优先 LLM 生成并经过只读 SQL 校验。
- `answer_analysis` 只基于 evidence 回答，partial evidence 会保留 gap。

当前仓库以后端 Python-only 为准。
