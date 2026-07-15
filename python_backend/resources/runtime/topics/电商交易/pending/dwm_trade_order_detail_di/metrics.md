# Metrics

- 订单明细优惠金额：SUM(discount_amt)，原始自然名=订单优惠金额，粒度=order_detail，单位=元，来源字段=discount_amt，同义词=订单明细优惠金额、discount_amt、按订单优惠金额、交易优惠金额，说明=订单明细中的优惠金额
- 订单明细下单商品数量：SUM(sku_cnt)，原始自然名=下单商品数量，粒度=order_detail，单位=单，来源字段=sku_cnt，同义词=订单明细下单商品数量、sku_cnt、订单下单商品数量、按订单下单商品数量、订单商品数量，说明=订单明细中的商品数量
- 订单明细商品总额分：SUM(product_amt)，原始自然名=商品总额分，粒度=order_detail，单位=元，来源字段=product_amt，同义词=订单明细商品总额分、product_amt、订单商品总额分、按订单商品总额分
- 订单明细运费金额分：SUM(freight_amt)，原始自然名=运费金额分，粒度=order_detail，单位=元，来源字段=freight_amt，同义词=订单明细运费金额分、freight_amt、订单运费金额分、按订单运费金额分
- 订单明细平台补贴金额：SUM(platform_subsidy_amt)，原始自然名=平台补贴金额，粒度=order_detail，单位=元，来源字段=platform_subsidy_amt，同义词=订单明细平台补贴金额、platform_subsidy_amt、订单平台补贴金额、按订单平台补贴金额
- 订单明细商家补贴金额：SUM(seller_subsidy_amt)，原始自然名=商家补贴金额，粒度=order_detail，单位=元，来源字段=seller_subsidy_amt，同义词=订单明细商家补贴金额、seller_subsidy_amt、订单商家补贴金额、按订单商家补贴金额
- 订单明细支付金额：SUM(pay_amt)，原始自然名=订单支付金额，粒度=order_detail，单位=元，来源字段=pay_amt，同义词=订单明细支付金额、pay_amt、按订单支付金额、交易支付金额、交易成交金额，说明=订单明细中的支付金额
