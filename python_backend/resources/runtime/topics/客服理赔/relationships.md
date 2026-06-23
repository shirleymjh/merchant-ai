# Topic Relationships: 客服理赔

## order_repay_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_cs_repay_detail_df
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 赔付单
- 用途: 赔付明细带出订单金额、商品和下单表现。

## ticket_repay_by_ticket
- 左表: dwm_cs_ticket_detail_di
- 右表: dwm_cs_repay_detail_df
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, ticket_id = ticket_id
- 粒度: 工单 x 赔付单
- 用途: 赔付明细带出工单状态、催单标识、订单号和商品信息。
