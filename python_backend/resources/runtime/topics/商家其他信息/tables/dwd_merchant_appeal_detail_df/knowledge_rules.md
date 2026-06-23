# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `merchant_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、merchant_id
- 明细查询优先使用明细表（Always Apply）：用户问“明细、哪几单、列表、记录”时，应使用该明细表返回真实单据，不要用 ads 指标表兜底伪装明细。，关键词=明细、哪几单、列表、记录
- 申诉与处罚口径（Always Apply）：申诉状态看 `appeal_status_name/code`；处罚类申诉需限定 `apply_type_name = '处罚'` 或 `apply_type_code = 6`，不要把所有申诉都当处罚。，关键词=申诉、处罚、appeal_status、apply_type
