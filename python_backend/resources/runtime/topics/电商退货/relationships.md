# Topic Relationships: 电商退货

## order_refund_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_trade_refund_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 退款单
- 用途: 退款明细带出原订单状态、订单金额、商品和下单时间。
- 注意: 一笔子订单可能有多笔退款，做订单金额汇总时必须先按子订单去重。
