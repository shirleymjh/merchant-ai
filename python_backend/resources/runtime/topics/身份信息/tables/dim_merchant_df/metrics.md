# 指标公式

- 商户入驻初始保证金元：SUM(init_deposit_amt)，单位=元，来源字段=init_deposit_amt，同义词=商户入驻初始保证金元、init_deposit_amt
- 保证金余额：SUM(deposit_amt)，单位=元，来源字段=deposit_amt，同义词=deposit_amt、保证金余额
- 商家数：COUNT(DISTINCT merchant_id)，单位=个，来源字段=merchant_id，同义词=商家数
- 冻结保证金：SUM(deposit_freeze)，单位=元，来源字段=deposit_freeze，同义词=冻结保证金
