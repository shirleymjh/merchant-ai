# Topic Relationships: 电商交易

## order_refund_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_trade_refund_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 退款单
- 用途: 订单明细带出对应退款单、退款原因、退款状态和退款金额。
- 注意: 汇总金额时要避免一单多退导致订单金额重复计数。

## order_ticket_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_cs_ticket_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 工单
- 用途: 订单明细带出对应工单、催单标识、工单状态和工单创建时间。

## order_repay_by_sub_order
- 左表: dwm_trade_order_detail_di
- 右表: dwm_cs_repay_detail_df
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, sub_order_id = sub_order_id
- 粒度: 子订单 x 赔付单
- 用途: 订单明细带出对应赔付单、赔付金额和赔付状态。

## goods_order_by_spu_id
- 左表: dwm_goods_detail_df
- 右表: dwm_trade_order_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, spu_id = spu_id
- 粒度: SPU x 子订单
- 用途: 商品审核/商品状态明细带出订单成交表现。

## merchant_order_by_user
- 左表: dim_merchant_df
- 右表: dwm_trade_order_detail_di
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家 x 子订单
- 用途: 商家维度信息带出订单明细。
