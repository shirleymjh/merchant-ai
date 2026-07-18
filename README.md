#Merchant AI

商家 AI 助手示例工程。当前在线查数采用 `单 Grounded DeepAgent Core + FileSystem-as-Context + Grounded Query Contract + Verified Artifact Ledger` 架构，包含 Python FastAPI 后端、Vue 前端、Doris 查询、MySQL 问答记录以及规则/记忆沉淀。

运行时基线为 **Python 3.11/3.12 + Deep Agents 0.6.12 + LangChain 1.x + LangGraph 1.x**。Python 3.9 和 LangChain 0.3 环境不再受支持；启动脚本检测到旧 `.venv` 时会保留旧目录，并自动使用 `.venv-deepagent`。

## 模块

- `python_backend`: Python FastAPI API，负责单 Core ReAct、受治理语义读取、确定性/Core SQL 查询、证据校验、Chat BI 回复、问答记录和每日经营日报。
- `frontend`: Vue 3 + Vite 商家助手页面，视觉参考美团企业版 AI 助手并替换为 yshopping 品牌。
- `python_backend/resources/runtime/rules`: 经过治理的平台规则文档，按标题和语义段落建立召回索引。
- `python_backend/resources/sql`: MySQL 初始化 SQL。
- `python_backend/resources/runtime/topics`: Topic、表、指标、字段、relationship 和查询约束等语义资产。

## Grounded Core 的边界

- DeepAgent Core 是唯一主 ReAct 循环。底层 graph 只承载通用 model ↔ tool 循环，不存在预编排的业务查询 workflow/DAG。
- Topic 由当前商家问题自动发现。Core 先看到 Topic L0 manifest 和一次 thin recall，再自行用 `ls/read_file/grep` 渐进读取表、指标、字段、relationship 与规则。
- 只有成功读取的精确语义内容可以形成 Grounded Query Contract；`ls/grep` 和召回排名只用于导航，不能授权指标口径或物理字段。
- Original Question Goal Contract 记录原问题必须覆盖的目标，但不选表、不写 SQL、不规定执行顺序；最终回答前必须通过 verified artifact coverage gate。
- 单表安全形状走确定性编译，包括单/多指标、简单 grouped/trend、TopN 和 entity lookup；JOIN、CTE、窗口函数与复杂依赖由同一个 Core 提交完整 SQL。
- 无依赖目标可由 Core 放入隔离 query branches，并发执行 Doris 与 Evidence 校验；依赖 verified entity set 的 entity chain 保持串行。查询 branch 不是 LLM subagent。
- 每个查询独立通过 Contract、generation/fingerprint、SQL AST、权限和 Evidence 校验，只把 verified artifact 合并回主 session。
- 一次运行共享 90 秒、LLM、工具和 Doris 预算，并记录各阶段耗时和按名称调用次数。预算耗尽时不会把未验证的部分结果包装成答案。
- Analysis Skill 当前只在查询与 Goal coverage 完成后隔离运行，不能参与取数或修改查询状态。

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

Python 版复用 `python_backend/resources` 和现有 Doris/MySQL 环境变量；DeepAgent 负责单 Core ReAct 与上下文承载，Grounded Runtime Kernel 负责 Contract 生命周期、SQL/权限校验、查询执行、Evidence 和 artifact ledger，不负责业务步骤编排。

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
