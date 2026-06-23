# Topic Relationships: 客服工单

## order_ticket_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_cs_ticket_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 工单
- 用途: 工单明细回看订单状态、商品、支付金额和发货时间。

## ticket_repay_by_ticket
- 左表: dwm_cs_ticket_detail_di
- 右表: dwm_cs_repay_detail_df
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, ticket_id = ticket_id
- 粒度: 工单 x 赔付单
- 用途: 工单明细带出赔付金额、赔付状态和到账状态。
