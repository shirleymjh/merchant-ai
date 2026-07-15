# Metrics

- 赔付明细金额：SUM(repay_amt)，原始自然名=赔付金额，粒度=repay_detail，单位=元，来源字段=repay_amt，同义词=赔付明细金额、repay_amt、赔付支付金额，说明=赔付/理赔单支付金额
- 赔付明细满多少元：SUM(coupon_rule_a)，原始自然名=满多少元，粒度=repay_detail，单位=元，来源字段=coupon_rule_a，同义词=赔付明细满多少元、coupon_rule_a、赔付满多少元
- 赔付明细减多少元：SUM(coupon_rule_b)，原始自然名=减多少元，粒度=repay_detail，单位=元，来源字段=coupon_rule_b，同义词=赔付明细减多少元、coupon_rule_b、赔付减多少元
- 赔付明细单量：COUNT(DISTINCT bill_id)，原始自然名=赔付单量，粒度=repay_detail，单位=单，来源字段=bill_id，同义词=赔付明细单量、repay_bill_cnt、赔付订单量、赔付订单数、理赔订单量、有没有赔付、有赔付订单、有产生赔付，说明=按赔付单号去重统计赔付单数量
- 商品维度赔付率：赔付单量 / 订单量，原始自然名=商品赔付率，粒度=repay_detail，单位=%，来源字段=repay_bill_cnt、order_detail_cnt，同义词=商品维度赔付率、compensation_rate，说明=同一商品或订单集合下，赔付单量占订单量的比例；查询时必须同时取赔付单量和订单量作为分子、分母
- 赔付明细打款成功赔付金额：SUM(CASE WHEN pay_status_name = '打款成功' THEN repay_amt ELSE 0 END)，原始自然名=打款成功赔付金额，粒度=repay_detail，单位=元，来源字段=pay_status_name、repay_amt，同义词=赔付明细打款成功赔付金额、repay_success_amt、赔付打款成功赔付金额，说明=只统计到账状态为打款成功的赔付金额
