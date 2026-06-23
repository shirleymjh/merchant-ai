# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `merchant_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、merchant_id
- 经营画像指标聚合（Always Apply）：ads_merchant_profile 是商家日粒度画像表；金额/数量类指标按天 SUM，比例/均值/评分/时长类指标按天 AVG 或按分子分母重算。，关键词=指标、聚合、画像、SUM、AVG
