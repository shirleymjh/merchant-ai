# Metrics

- 保证金明细充值金额：SUM(deposit_recharge_amt)，原始自然名=保证金充值金额，粒度=merchant_deposit_detail，单位=元，来源字段=deposit_recharge_amt，同义词=保证金明细充值金额、deposit_recharge_amt、金额, 通用单位元、保证金补缴金额、保证金缴纳金额，说明=金额, 通用单位元
- 保证金明细充值流水数：COUNT(DISTINCT deposit_recharge_id)，原始自然名=保证金充值流水数，粒度=merchant_deposit_detail，单位=笔，来源字段=deposit_recharge_id，同义词=保证金明细充值流水数、deposit_recharge_cnt，说明=按保证金补缴/充值申请号去重统计流水数
