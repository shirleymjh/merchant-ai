# Metrics

- 商户入驻初始保证金元：SUM(init_deposit_amt)，原始自然名=商户入驻初始保证金元，粒度=merchant_dimension，单位=元，来源字段=init_deposit_amt，同义词=商户入驻初始保证金元、init_deposit_amt
- 保证金余额：SUM(deposit_amt)，原始自然名=保证金余额，粒度=merchant_dimension，单位=元，来源字段=deposit_amt，同义词=保证金余额、deposit_amt，说明=保证金
- 商家数：COUNT(DISTINCT merchant_id)，原始自然名=商家数，粒度=merchant_dimension，单位=个，来源字段=merchant_id，同义词=商家数、merchant_cnt，说明=按商家 ID 去重统计商家数量
- 冻结保证金：SUM(deposit_freeze)，原始自然名=冻结保证金，粒度=merchant_dimension，单位=元，来源字段=deposit_freeze，同义词=冻结保证金、merchant_deposit_freeze_amt，说明=商家当前冻结保证金金额
