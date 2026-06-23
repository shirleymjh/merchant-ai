# Topic Relationships: 经营画像

## profile_order_by_user
- 左表: ads_merchant_profile
- 右表: dwm_trade_order_detail_di
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家日期 x 子订单
- 用途: merchant_id 先约束经营画像，再通过 merchant_id 映射订单明细 seller_id。

## profile_refund_by_user
- 左表: ads_merchant_profile
- 右表: dwm_trade_refund_detail_di
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家日期 x 退款单
- 用途: merchant_id 先约束经营画像，再通过 merchant_id 映射退款明细 seller_id。

## profile_ticket_by_user
- 左表: ads_merchant_profile
- 右表: dwm_cs_ticket_detail_di
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家日期 x 工单
- 用途: merchant_id 先约束经营画像，再通过 merchant_id 映射客服工单 seller_id。

## profile_repay_by_user
- 左表: ads_merchant_profile
- 右表: dwm_cs_repay_detail_df
- JOIN: LEFT JOIN
- Key: merchant_id = seller_id
- 粒度: 商家日期 x 赔付单
- 用途: merchant_id 先约束经营画像，再通过 merchant_id 映射赔付明细 seller_id。
