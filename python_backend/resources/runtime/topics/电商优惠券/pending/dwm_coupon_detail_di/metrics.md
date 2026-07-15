# Metrics

- 优惠券明细金额：SUM(coupon_amt)，原始自然名=优惠券金额，粒度=coupon_detail，单位=元，来源字段=coupon_amt，同义词=优惠券明细金额、coupon_amt、优惠券面额、券金额投入、优惠券金额投入、优惠券投入、券活动投入、优惠券活动投入，说明=优惠券金额投入，按优惠券明细表 coupon_amt 聚合；可用于回答券金额投入、优惠券投入、优惠投入等问题
- 优惠券明细津贴金额：SUM(CAST(allowance_amt AS DECIMAL(18,2)))，原始自然名=津贴金额，粒度=coupon_detail，单位=元，来源字段=allowance_amt，同义词=优惠券明细津贴金额、allowance_amt、优惠券津贴金额、券明细津贴金额
- 优惠券发放明细量：COUNT(DISTINCT coupon_id)，原始自然名=优惠券发放量，粒度=coupon_detail，单位=张，来源字段=coupon_id，同义词=优惠券发放明细量、coupon_issue_cnt、优惠券明细发放量、优惠券发放量、优惠券发放数、券发放量、券发放数，说明=按券编号去重统计优惠券发放量
- 优惠券领取明细量：SUM(CASE WHEN is_receive = 1 THEN 1 ELSE 0 END)，原始自然名=优惠券领取量，粒度=coupon_detail，单位=张，来源字段=is_receive，同义词=优惠券领取明细量、coupon_receive_cnt、优惠券明细领取量、优惠券使用量、优惠券使用数、优惠券领取量、优惠券领取数、券使用量、券领取量，说明=按是否抢到优惠券统计领取量
- 优惠券退回明细量：COUNT(DISTINCT CASE WHEN coupon_refund_id <> '' THEN coupon_refund_id END)，原始自然名=优惠券退回量，粒度=coupon_detail，单位=张，来源字段=coupon_refund_id，同义词=优惠券退回明细量、coupon_refund_cnt、优惠券明细退回量、优惠券退回量、优惠券退回数、券退回量，说明=按优惠券退回编号统计退回量
