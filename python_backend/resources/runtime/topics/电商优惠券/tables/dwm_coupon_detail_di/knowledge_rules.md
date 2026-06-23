# 业务规则

- 时间范围过滤（Always Apply）：涉及最近N天、昨天、上个月等时间表达时，必须优先使用 `pt` 做业务日期过滤。，关键词=时间、最近、pt
- 商家范围过滤（Always Apply）：回答单个商家问题时，必须用 `seller_id` 过滤当前商家，避免查到全站数据。，关键词=商家、过滤、seller_id
- 明细查询优先使用明细表（Always Apply）：用户问“明细、哪几单、列表、记录”时，应使用该明细表返回真实单据，不要用 ads 指标表兜底伪装明细。，关键词=明细、哪几单、列表、记录
- 优惠券与优惠订单区分（Always Apply）：dwm_coupon_detail_di 是券/津贴明细表，可回答优惠券发放、领取、退回和优惠金额；用户问“优惠订单量/优惠订单数”时，应优先使用经营画像中的优惠订单指标，不要用券数量替代订单量。，关键词=优惠券、优惠订单量、优惠订单数、券、津贴
- 优惠券领取与退回口径（Always Apply）：优惠券领取看 `is_receive = 1`；优惠券退回看 `coupon_refund_id` 或 `refund_time`；津贴金额字段 `allowance_amt` 为字符串时需 CAST 后聚合。，关键词=领取、退回、is_receive、allowance_amt
