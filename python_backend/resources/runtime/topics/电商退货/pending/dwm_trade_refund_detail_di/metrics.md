# Metrics

- 退款明细商品数量：SUM(sku_count)，原始自然名=退款商品数量，粒度=refund_detail，单位=单，来源字段=sku_count，同义词=退款明细商品数量、sku_count、售后商品数量，说明=退款/售后明细中的商品数量
- 退款明细退款金额：SUM(pay_amt)，原始自然名=退款金额，粒度=refund_detail，单位=元，来源字段=pay_amt，同义词=退款明细退款金额、pay_amt、退款金额、退货金额、退款额、售后金额、refund_amt、refund_amount，说明=退款明细中的退款金额，适用于按订单、商品、日期等明细维度统计退款金额。
- 退款明细优惠金额：SUM(refund_discount_amt)，原始自然名=退款优惠金额，粒度=refund_detail，单位=元，来源字段=refund_discount_amt，同义词=退款明细优惠金额、refund_discount_amt、售后优惠金额，说明=退款/售后明细中的优惠金额
- 退款明细物流金额：SUM(logistic_amt)，原始自然名=物流金额，粒度=refund_detail，单位=元，来源字段=logistic_amt，同义词=退款明细物流金额、logistic_amt、退款物流金额
- 退款明细单量：COUNT(DISTINCT refund_id)，原始自然名=退款单量，粒度=refund_detail，单位=单，来源字段=refund_id，同义词=退款明细单量、refund_bill_cnt、退款订单量、有退款的订单、有退款的订单数、有退货的订单、有退货的订单数、发生退款的订单、发生退款的订单数、发生退货的订单、发生退货的订单数、退货订单数、退款高发商品，说明=按退款单号去重统计退款/售后单数量
- 商品维度退货率口径说明：，原始自然名=商品退货率，粒度=refund_detail，单位=%，来源字段=refund_id，同义词=商品维度退货率口径说明、商品退货率口径、按商品退货率口径，说明=商品维度退货率需要同商品粒度的退货量和订单量作为分子分母；当前退货明细表只提供退货侧分子，不能单表直接产出完整退货率。
