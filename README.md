# yshopping Merchant AI

商家 AI 助手示例工程，包含 Spring Boot 后端、Vue 前端、Doris 查询、MySQL 问答记录、规则/wiki 记忆和 Java 状态图问答流程。

## 模块

- `backend`: Java Spring Boot API，负责意图识别、Doris 查询、LLM 回复、问答记录和每日经营日报。
- `frontend`: Vue 3 + Vite 商家助手页面，视觉参考美团企业版 AI 助手并替换为 yshopping 品牌。
- `backend/src/main/resources/wiki`: 平台规则和可复用 LLM wiki 记忆，人工可直接补充 Markdown。
- `backend/src/main/resources/sql`: MySQL 初始化 SQL。

## 关键环境变量

```bash
export YSHOPPING_LLM_BASE_URL="https://way.ydata.vip/v1"
export YSHOPPING_LLM_MODEL="gpt-5.5"
export YSHOPPING_LLM_API_KEY="替换为你的本地密钥"
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

后端需要 JDK 17+ 和 Maven：

```bash
cd backend
mvn spring-boot:run
```

前端：

```bash
cd frontend
npm install
npm run dev
```

