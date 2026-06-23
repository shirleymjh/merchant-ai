# Topic Relationships: 身份信息

## merchant_order_by_user
- 左表: dim_merchant_df
- 右表: dwm_trade_order_detail_di
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家 x 子订单
- 用途: 商家维度信息带出订单明细。
