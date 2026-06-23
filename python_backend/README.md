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
