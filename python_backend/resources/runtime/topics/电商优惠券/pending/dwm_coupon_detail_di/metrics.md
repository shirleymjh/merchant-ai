# 指标公式

- 优惠券金额：SUM(coupon_amt)，单位=元，来源字段=coupon_amt，同义词=优惠券金额、优惠券面额、券面额、券金额、券金额投入、优惠券金额投入、优惠券投入、优惠投入、券投入、coupon_amt
- 津贴金额：SUM(CAST(allowance_amt AS DECIMAL(18,2)))，单位=元，来源字段=allowance_amt，同义词=津贴金额、allowance_amt
- 优惠券发放量：COUNT(DISTINCT coupon_id)，单位=张，来源字段=coupon_id，同义词=优惠券发放量
- 优惠券领取量：SUM(CASE WHEN is_receive = 1 THEN 1 ELSE 0 END)，单位=张，来源字段=is_receive，同义词=优惠券领取量
- 优惠券退回量：COUNT(DISTINCT CASE WHEN coupon_refund_id <> '' THEN coupon_refund_id END)，单位=张，来源字段=coupon_refund_id，同义词=优惠券退回量
