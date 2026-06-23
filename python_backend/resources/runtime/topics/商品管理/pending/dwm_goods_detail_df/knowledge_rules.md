# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `seller_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、seller_id
- 明细查询优先使用明细表（Always Apply）：用户问“明细、哪几单、列表、记录”时，应使用该明细表返回真实单据，不要用 ads 指标表兜底伪装明细。，关键词=明细、哪几单、列表、记录
- 商品审核明细口径（Always Apply）：用户问商品审核拒绝/通过明细时，应使用 dwm_goods_detail_df；审核拒绝按 `is_audit_pass = 0`，审核通过按 `is_audit_pass = 1`，上架商品按 `spu_status_name = '上架'` 或 `spu_status_code = 1`。，关键词=商品、审核、拒绝、通过、上架
- 商品价格单位（Always Apply）：`spu_auth_price` 为发售价，单位是分；展示给商家时如需金额应除以 100 转元。，关键词=价格、售价、spu_auth_price、分
