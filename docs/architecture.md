# yshopping 商家 AI 助手架构

## 问答流程

```mermaid
flowchart LR
  A["用户输入"] --> B["读取历史 LLM wiki 记忆"]
  B --> C["用户意图识别"]
  C --> D{"问题分类"}
  D --> R["平台规则: rule.md"]
  D --> T["电商交易"]
  D --> F["电商退货"]
  D --> S["电商客服工单"]
  D --> P["电商理赔/赔付"]
  D --> CPN["电商优惠券"]
  D --> G["商品管理"]
  D --> M["商家其他信息"]
  D --> I["身份信息"]
  T --> Q["指标: ads_merchant_profile; 明细: dwm_trade_order_detail_di"]
  F --> Q2["指标: ads_merchant_profile; 明细: dwm_trade_refund_detail_di"]
  S --> Q3["指标: ads_merchant_profile; 明细: dwm_cs_ticket_detail_di"]
  P --> Q4["指标: ads_merchant_profile; 明细: dwm_cs_repay_detail_df"]
  CPN --> Q5["指标: ads_merchant_profile; 明细: dwm_coupon_detail_di"]
  G --> Q6["指标: ads_merchant_profile; 明细: dwm_goods_detail_df"]
  M --> Q7["指标: ads_merchant_profile; 明细: appeal/deposit 表"]
  I --> Q8["dim_merchant_df"]
  R --> O["整理自然语言话术"]
  Q --> O
  Q2 --> O
  Q3 --> O
  Q4 --> O
  Q5 --> O
  Q6 --> O
  Q7 --> O
  Q8 --> O
  O --> W["写入 MySQL merchant_ai_answer"]
```

## 数据落库

`merchant_ai_answer` 字段：

- `id`: 每一次对话信息 id。
- `question`: 用户提问信息。
- `answer`: 模型回复信息。
- `is_adopted`: 用户是否点击采纳。
- `like_flag`: 点赞。
- `dislike_flag`: 点踩。
- `merchant_id`: 商家 id。
- `merchant_name`: 商家名称。
- `question_category_name`: 问题分类名称。
- `doris_tables`: 调用的 Doris 数据表。
- `suggested_questions`: 猜你想问。
- `create_time`: 创建时间。
- `modify_time`: 模型回复后的变更时间。

## 特殊策略

- 打招呼类输入只自然回复，不写入 `merchant_ai_answer`。
- 无效意图回复“请提工单进行人工咨询”，不写入 `merchant_ai_answer`。
- 当前本地商家 id 默认为 `100`。
- 最近 N 天且 N 大于 1 的指标按 `pt` 每日汇总。
- 明细类问题按商家/卖家 id 过滤，默认最多返回 20 条关键记录。
- 每日 10 点从 `ads_merchant_profile` 读取昨日数据并生成两条经营建议。

