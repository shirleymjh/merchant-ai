# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `merchant_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、merchant_id
- 明细查询优先使用明细表（Always Apply）：用户问“明细、哪几单、列表、记录”时，应使用该明细表返回真实单据，不要用 ads 指标表兜底伪装明细。，关键词=明细、哪几单、列表、记录
- 供应链履约口径（Always Apply）：dwm_scm_detail_di 覆盖入库、质检、鉴定、出库全流程；发货超时明细按 `outbound_modify_time > outbound_latest_time` 判断，经营画像中的发货超时指标只适合汇总趋势。，关键词=履约、发货超时、出库、质检、鉴定
