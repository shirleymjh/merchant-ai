# 指标公式

- 退款商品数量：SUM(sku_count)，单位=单，来源字段=sku_count，同义词=退款商品数量、售后商品数量、sku_count
- 退款关联支付金额：SUM(pay_amt)，单位=元，来源字段=pay_amt，同义词=退款关联支付金额、退款订单支付金额、售后关联支付金额、pay_amt
- 退款优惠金额：SUM(refund_discount_amt)，单位=元，来源字段=refund_discount_amt，同义词=退款优惠金额、售后优惠金额、refund_discount_amt
- 物流金额：SUM(logistic_amt)，单位=元，来源字段=logistic_amt，同义词=物流金额、logistic_amt
- 退款单量：COUNT(DISTINCT refund_id)，单位=单，来源字段=refund_id，同义词=退款单量、退款量、退款订单量、退款单、售后单量
- 商品退货率：退货量 / 订单量，单位=%，来源指标=refund_bill_cnt、order_detail_cnt，同义词=商品退款率、退款率、退货率、售后率、refund_rate、return_rate
