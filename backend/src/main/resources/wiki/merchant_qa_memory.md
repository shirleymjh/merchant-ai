# yshopping 商家 AI 助手可复用记忆

## 分类

- 平台商家规则：读取 `rule.md` 回答培训、使用规则、平台政策。
- 电商交易：指标取 `ads_merchant_profile`，明细取 `dwm_trade_order_detail_di`。
- 电商退货：指标取 `ads_merchant_profile`，明细取 `dwm_trade_refund_detail_di`。
- 电商客服工单：指标取 `ads_merchant_profile`，明细取 `dwm_cs_ticket_detail_di`，对话内容可扩展 `ads_risk_csr_ticket_all_content`。
- 电商理赔/赔付：指标取 `ads_merchant_profile`，明细取 `dwm_cs_repay_detail_df`。
- 电商优惠券：指标取 `ads_merchant_profile`，明细取 `dwm_coupon_detail_di`。
- 商品管理：指标取 `ads_merchant_profile`，明细取 `dwm_goods_detail_df`。
- 商家其他信息：保证金、处罚、申诉等指标取 `ads_merchant_profile`，明细取 `dwd_merchant_appeal_detail_df`、`dwd_merchant_deposit_recharge_df`。
- 身份信息：读取 `dim_merchant_df`。

## 回答口径

- 最近 N 天且 N 大于 1 的指标，按 `pt` 汇总成每日数据后再总结趋势。
- 单日指标直接读取对应 `pt` 的画像字段。
- 明细类问题最多返回 20 条关键记录，不直接暴露无关字段。
- 规则类问题优先回答用户明确提到的子主题，不要只返回泛化的大类话术。
- 如果用户问“类目规范”“资质要求”“审核拒绝原因”，应围绕该子主题先直接作答，再补充上架通用建议。
- 如果用户问“工单进度”“退款怎么处理”“物流单号关联订单”，应优先说明处理路径、关键字段和排查步骤。
- 日常打招呼自然回复，不写入问答记录。
- 无效意图返回人工工单提示，不写入问答记录。

## 高频规则测试问法

- 商品上架要求是什么
- 商品类目规范是什么
- 商品类目怎么选
- 商品审核被拒怎么办
- 发货超时怎么处理
- 物流单号怎么关联订单
- 退款处理规则是什么
- 卖家责任退货怎么判断
- 工单进度怎么查
- 工单二次开启是什么意思
- 赔付原因怎么查
- 优惠券规则有哪些
- 保证金余额怎么看
- 申诉流程是什么
