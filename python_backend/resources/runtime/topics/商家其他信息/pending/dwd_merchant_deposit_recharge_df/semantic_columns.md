# 商家其他信息 / dwd_merchant_deposit_recharge_df

状态：PENDING_REVIEW
表说明：dwd-商家域-商家保证金充值表
数据粒度：保证金充值流水粒度
时间字段：`pt`
商家过滤字段：`merchant_id`
人工业务说明：商家保证金充值流水表，承载保证金充值/补缴流水与金额

## 列级语义
- `create_time`：创建时间，角色=TIME，说明=创建时间，同义词=创建时间、create_time
- `modify_time`：修改时间，角色=TIME，说明=修改时间，同义词=修改时间、modify_time
- `merchant_id`：商家id，角色=KEY，说明=商家id，同义词=商家id、merchant_id
- `user_id`：用户id，角色=KEY，说明=用户id，同义词=用户id、user_id
- `deposit_recharge_id`：补缴单单号(充值申请号)，角色=KEY，说明=补缴单单号(充值申请号)，同义词=补缴单单号(充值申请号)、deposit_recharge_id、保证金、押金
- `trans_id`：交易流水号，角色=KEY，说明=交易流水号，同义词=交易流水号、trans_id
- `currency`：币种，角色=OTHER，说明=币种，同义词=币种、currency
- `deposit_recharge_amt`：金额，角色=METRIC，说明=金额, 通用单位元，公式=SUM(deposit_recharge_amt)，同义词=金额、deposit_recharge_amt、金额, 通用单位元、保证金、押金
- `trans_voucher`：交易流水凭证，角色=OTHER，说明=交易流水凭证，同义词=交易流水凭证、trans_voucher
- `remark`：备注，角色=OTHER，说明=备注，同义词=备注、remark
- `pt`：日期分区yyyyMMdd，角色=TIME，说明=日期分区yyyyMMdd，同义词=日期分区yyyyMMdd、pt
