#Merchant AI

商家 AI 助手示例工程，包含 Python FastAPI + LangGraph 后端、Vue 前端、Doris 查询、MySQL 问答记录、规则/记忆沉淀，以及 `Main Agent ReAct Runtime + QueryGraph + NodeWorker` 的 Agent Harness 问答流程。

## 模块

- `python_backend`: Python FastAPI API，负责 Main Agent 编排、LangGraph 状态图、业务范围澄清、记忆召回、Doris 查询、Chat BI 回复、问答记录和每日经营日报。
- `frontend`: Vue 3 + Vite 商家助手页面，视觉参考美团企业版 AI 助手并替换为 yshopping 品牌。
- `python_backend/resources/runtime/rules`: 经过治理的平台规则文档，按标题和语义段落建立召回索引。
- `python_backend/resources/sql`: MySQL 初始化 SQL。
- `python_backend/resources/runtime/topics`: Topic 资产、语义层、relationship 和 QueryGraph 规划资产。

## 关键环境变量

```bash
export YSHOPPING_LLM_BASE_URL="https://way.ydata.vip/v1"
export YSHOPPING_LLM_MODEL="gpt-5.5"
export YSHOPPING_LLM_API_KEY="替换为你的本地密钥"
export YSHOPPING_PREFLIGHT_LLM_BASE_URL="https://小模型服务/v1"
export YSHOPPING_PREFLIGHT_SEMANTIC_ROUTE_MODEL="轻量路由小模型"
export YSHOPPING_PREFLIGHT_LLM_API_KEY="替换为轻量路由密钥"
export YSHOPPING_EMBEDDING_BASE_URL="https://api.openai.com/v1"
export YSHOPPING_EMBEDDING_MODEL="text-embedding-3-small"
export YSHOPPING_EMBEDDING_API_KEY="替换为你的 embedding 密钥"
export YSHOPPING_ES_ENABLED="true"
export YSHOPPING_ES_BASE_URL="http://127.0.0.1:9200"
export YSHOPPING_ES_INDEX="merchant_ai_recall"
export YSHOPPING_MERCHANT_ID="100"

export YSHOPPING_DORIS_JDBC_URL="jdbc:mysql://127.0.0.1:9030/yshopping?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai"
export YSHOPPING_DORIS_USERNAME="root"
export YSHOPPING_DORIS_PASSWORD=""

export YSHOPPING_ANSWER_JDBC_URL="jdbc:mysql://127.0.0.1:3306/yshopping?useUnicode=true&characterEncoding=utf8&useSSL=false&serverTimezone=Asia/Shanghai"
export YSHOPPING_ANSWER_USERNAME="root"
export YSHOPPING_ANSWER_PASSWORD=""
```

如果本地暂时没有独立 MySQL，可先把 `YSHOPPING_ANSWER_JDBC_URL` 指到 Doris 的 MySQL 端口做联调，但生产建议独立 MySQL 保存问答交互记录。

## 启动

后端：

```bash
bash scripts/start_python_backend.sh
```

或手动启动：

```bash
cd python_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

Python 版复用 `python_backend/resources` 和现有 Doris/MySQL 环境变量；核心问答流程由 LangGraph `StateGraph` 编排 V2 Main Agent ReAct / QueryGraph / NodeWorker 节点。

## ES Recall

本项目支持将“意图识别前的混合检索”切到 Elasticsearch。后端提供两个接口：

- `GET /api/es/recall-mapping`：查看推荐的 recall index mapping
- `POST /api/es/rebuild-recall-index?merchantId=100`：重建 recall 索引并灌入受治理规则与 Topic 语义资产

也可以直接执行脚本：

```bash
bash scripts/rebuild_recall_index.sh
```

## Golden Evaluation

运营接口 `POST /api/ops/golden-evaluations` 会按以下层次定位失败：

`recall -> queryGraph -> sql -> execution -> evidence -> answer`

用例位于 `python_backend/resources/evaluation/golden_cases.jsonl`。给用例增加 `expectedResult` 后，评测会执行只读标准 SQL，并比较标准结果与 Agent 实际查询结果：

```json
{
  "expectedResult": {
    "sql": "SELECT SUM(metric) AS metric FROM table WHERE merchant_id = %s",
    "params": ["$merchant_id"],
    "columns": ["metric"],
    "keyColumns": [],
    "orderSensitive": false,
    "numericTolerance": 0.01
  }
}
```

报告中的 `executionAccuracy` 只统计配置了标准 SQL 或标准结果的用例，`firstFailureBreakdown` 用于判断问题最早发生在召回、规划、SQL、结果、证据还是回答阶段。
