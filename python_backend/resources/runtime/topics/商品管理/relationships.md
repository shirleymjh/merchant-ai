# Topic Relationships: 商品管理

## goods_order_by_spu_id
- 左表: dwm_goods_detail_df
- 右表: dwm_trade_order_detail_di
- JOIN: LEFT JOIN
- Key: seller_id = seller_id, spu_id = spu_id
- 粒度: SPU x 子订单
- 用途: 商品审核/商品状态明细带出订单成交明细。
