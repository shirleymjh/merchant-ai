# 指标公式

- 赔付金额：SUM(repay_amt)，单位=元，来源字段=repay_amt，同义词=赔付金额、理赔金额、赔付支付金额、repay_amt
- 满多少元：SUM(coupon_rule_a)，单位=元，来源字段=coupon_rule_a，同义词=满多少元、coupon_rule_a
- 减多少元：SUM(coupon_rule_b)，单位=元，来源字段=coupon_rule_b，同义词=减多少元、coupon_rule_b
- 赔付单量：COUNT(DISTINCT bill_id)，单位=单，来源字段=bill_id，同义词=赔付单量
- 打款成功赔付金额：SUM(CASE WHEN pay_status_name = '打款成功' THEN repay_amt ELSE 0 END)，单位=元，来源字段=pay_status_name、repay_amt，同义词=打款成功赔付金额
- 工单量：COUNT(DISTINCT ticket_id)，单位=单，来源字段=ticket_id，同义词=工单量
