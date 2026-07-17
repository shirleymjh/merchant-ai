#Merchant AI

商家 AI 助手示例工程，采用 Diana 式 `Deep Agent Lead Runtime + FileSystem-as-Context + QueryGraph + NodeWorker` 架构，包含 Python FastAPI 后端、Vue 前端、Doris 查询、MySQL 问答记录以及规则/记忆沉淀。

运行时基线为 **Python 3.11/3.12 + Deep Agents 0.6.12 + LangChain 1.x + LangGraph 1.x**。Python 3.9 和 LangChain 0.3 环境不再受支持；启动脚本检测到旧 `.venv` 时会保留旧目录，并自动使用 `.venv-deepagent`。

## 模块

- `python_backend`: Python FastAPI API，负责 Main Agent 编排、LangGraph 状态图、业务范围澄清、记忆召回、Doris 查询、Chat BI 回复、问答记录和每日经营日报。
- `frontend`: Vue 3 + Vite 商家助手页面，视觉参考美团企业版 AI 助手并替换为 yshopping 品牌。
- `python_backend/resources/runtime/rules`: 经过治理的平台规则文档，按标题和语义段落建立召回索引。
- `python_backend/resources/sql`: MySQL 初始化 SQL。
- `python_backend/resources/runtime/topics`: Topic 资产、语义层、relationship 和 QueryGraph 规划资产。

## DeepAgent 与 Diana 的边界

- DeepAgent Core 是唯一主 ReAct 循环，默认最多 16 个受治理动作；Planner 不是 subagent，也没有隐藏文件工具循环。
- Topic 由当前商家问题自动发现。Core 首先只看到跨候选 Topic 的 L0 表清单和业务摘要，再自行用 `ls/read_file/grep` 读取表详情、指标/字段/schema、关系与业务规则。
- 只有成功 `read_file` 的精确内容进入 Planner 可信证据账本；`ls/grep` 和 RAG 排名只用于导航，不能直接授权指标口径或物理字段。
- DeepAgent 托管短期消息上下文、checkpoint、自动摘要、state filesystem 和超过 20k token 的 tool result offload。
- Diana domain 生成的 QueryGraph、SQL/result 与证据中间产物以只读 `/artifacts` 挂载给 DeepAgent，Core 可按需继续 `ls/read_file/grep`，不会把整份结果重新注入 prompt。
- 商家个人 memory（偏好、习惯、近期关注）继续由 Diana 的租户隔离/TTL/冲突/反馈机制自动沉淀和按题召回；低可信内容只会被自动隔离，敏感标识拒绝落库，不会创建人审队列。共享指标口径、规则与术语才走独立的 knowledge 确认/审核/发布流程；私有或未标 scope 的候选禁止进入共享发布链路。两者都不使用全量常驻、可直接编辑的 `AGENTS.md` 代替。
- 执行产生零行、SQL 错误、新鲜度或 snapshot 缺口时，Core 会看到结构化 observation，并可在预算内补读、换表、重新规划和重试；空结果不会被解释为业务为 0。

## 关键环境变量

```bash
export YSHOPPING_LLM_BASE_URL="https://way.ydata.vip/v1"
export YSHOPPING_LLM_MODEL="gpt-5.5"
export YSHOPPING_LLM_API_KEY="替换为你的本地密钥"
export YSHOPPING_AGENT_PLANNER_PROMPT_BUDGET_CHARS="30000"
export YSHOPPING_AGENT_LEAD_ACTION_RETRIES="1"
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
python3.11 -m venv .venv-deepagent
source .venv-deepagent/bin/activate
pip install -e .
python -m merchant_ai.runtime_compat
uvicorn app.main:app --host 0.0.0.0 --port 8088 --reload
```

前端：

```bash
cd frontend
npm install
npm run dev
```

Python 版复用 `python_backend/resources` 和现有 Doris/MySQL 环境变量；DeepAgent 负责 Core ReAct 与上下文承载，现有 LangGraph domain kernel 作为受治理工具执行 QueryGraph、NodeWorker、证据和回答动作。

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
