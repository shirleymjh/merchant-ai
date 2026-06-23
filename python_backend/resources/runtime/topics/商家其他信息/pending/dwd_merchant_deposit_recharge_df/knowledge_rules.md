# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `merchant_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、merchant_id
- 保证金充值流水口径（Always Apply）：用户问保证金充值、补缴、缴纳流水或充值金额时，必须查询 dwd_merchant_deposit_recharge_df；经营画像中的 deposit_pay_cnt_1d 只能作为次数指标，不能替代充值明细或金额。，关键词=保证金、充值、补缴、流水、deposit_recharge_amt
