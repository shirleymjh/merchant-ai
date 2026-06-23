# 经营画像 / ads_merchant_profile

状态：PENDING_REVIEW
数据粒度：商家-日期粒度经营指标画像
时间字段：`pt`
商家过滤字段：`merchant_id`
人工业务说明：商家日粒度经营指标画像，承载订单、GMV、退款、工单、履约、商品、优惠、保证金等核心指标

## 列级语义
- `merchant_id`：商家id，角色=KEY，说明=商家id，同义词=商家id、merchant_id
- `pt`：业务日期，角色=TIME，说明=业务日期，同义词=业务日期、pt
- `user_id`：用户id，角色=KEY，说明=用户id，同义词=用户id、user_id
- `merchant_type_name`：商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6，角色=DIMENSION，说明=商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方，同义词=商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6、merchant_type_name、商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方
- `brand_type_name`：资质类型name，角色=DIMENSION，说明=资质类型name -1代表商家是服务商 0.基本信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商，同义词=资质类型name、brand_type_name、资质类型name -1代表商家是服务商 0.基本信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商
- `balance_type_name`：结算类型name 0.实时 1.周结 2.半月结 3.月结，角色=DIMENSION，说明=结算类型name 0.实时 1.周结 2.半月结 3.月结，同义词=结算类型name 0.实时 1.周结 2.半月结 3.月结、balance_type_name
- `mobile`：商家手机号，角色=OTHER，说明=商家手机号，同义词=商家手机号、mobile
- `company_name`：公司名称，角色=DIMENSION，说明=公司名称，同义词=公司名称、company_name
- `license_id`：营业执照编号，角色=KEY，说明=营业执照编号，同义词=营业执照编号、license_id
- `is_unconditional_refund`：是否支持七天无理由退货 0否 1是，角色=DIMENSION，说明=是否支持七天无理由退货 0否 1是，同义词=是否支持七天无理由退货 0否 1是、is_unconditional_refund、退款、售后
- `is_invoice`：是否支持开具发票 0否 1是，角色=DIMENSION，说明=是否支持开具发票 0否 1是，同义词=是否支持开具发票 0否 1是、is_invoice
- `refund_mobile`：退货地址收货人手机号，角色=OTHER，说明=退货地址收货人手机号，同义词=退货地址收货人手机号、refund_mobile、退款、售后
- `currency`：币种，角色=OTHER，说明=币种，同义词=币种、currency
- `contact_name`：联系人姓名，角色=DIMENSION，说明=联系人姓名，同义词=联系人姓名、contact_name
- `contact_idcard`：联系人身份证号，角色=OTHER，说明=联系人身份证号，同义词=联系人身份证号、contact_idcard
- `business_address`：经营商详细地址，角色=OTHER，说明=经营商详细地址，同义词=经营商详细地址、business_address
- `send_address`：发货详细地址，角色=OTHER，说明=发货详细地址，同义词=发货详细地址、send_address
- `refnd_address`：退货地址详细地址，角色=OTHER，说明=退货地址详细地址，同义词=退货地址详细地址、refnd_address
- `bank_name`：开户行，角色=DIMENSION，说明=开户行，同义词=开户行、bank_name
- `bank_account`：银行账号，角色=OTHER，说明=银行账号，同义词=银行账号、bank_account
- `account_type_name`：账户类型name 1.对公 2.对私，角色=DIMENSION，说明=账户类型name 1.对公 2.对私，同义词=账户类型name 1.对公 2.对私、account_type_name
- `poundage_discount`：费率折扣，角色=OTHER，说明=费率折扣，同义词=费率折扣、poundage_discount
- `deposit_amt`：保证金余额元，角色=METRIC，说明=保证金余额元，公式=SUM(deposit_amt)，同义词=保证金余额元、deposit_amt
- `order_cnt_1d`：总订单量，角色=METRIC，说明=总订单量-最近1天，公式=SUM(order_cnt_1d)，同义词=总订单量、order_cnt_1d、总订单量-最近1天、订单
- `order_user_cnt_1d`：下单用户量，角色=METRIC，说明=下单用户量-最近1天，公式=SUM(order_user_cnt_1d)，同义词=下单用户量、order_user_cnt_1d、下单用户量-最近1天、订单
- `order_gmv_amt_1d`：总gmv金额元，角色=METRIC，说明=总gmv金额元-最近1天，公式=SUM(order_gmv_amt_1d)，同义词=总gmv金额元、order_gmv_amt_1d、总gmv金额元-最近1天、GMV、成交额、销售额、交易流水、订单
- `pay_order_cnt_1d`：支付订单量，角色=METRIC，说明=支付订单量-最近1天，公式=SUM(pay_order_cnt_1d)，同义词=支付订单量、pay_order_cnt_1d、支付订单量-最近1天、订单
- `pay_gmv_amt_1d`：支付gmv金额元，角色=METRIC，说明=支付gmv金额元-最近1天，公式=SUM(pay_gmv_amt_1d)，同义词=支付gmv金额元、pay_gmv_amt_1d、支付gmv金额元-最近1天、GMV、成交额、销售额、交易流水
- `trade_success_level1_category_cnt_1d`：交易成功一级类目数，角色=DIMENSION，说明=交易成功一级类目数-最近1天，同义词=交易成功一级类目数、trade_success_level1_category_cnt_1d、交易成功一级类目数-最近1天
- `trade_success_level2_category_cnt_1d`：交易成功二级类目数，角色=DIMENSION，说明=交易成功二级类目数-最近1天，同义词=交易成功二级类目数、trade_success_level2_category_cnt_1d、交易成功二级类目数-最近1天
- `trade_success_level3_category_cnt_1d`：交易成功三级类目数，角色=DIMENSION，说明=交易成功三级类目数-最近1天，同义词=交易成功三级类目数、trade_success_level3_category_cnt_1d、交易成功三级类目数-最近1天
- `trade_success_order_cnt_1d`：交易成功订单量，角色=METRIC，说明=交易成功订单量-最近1天，公式=SUM(trade_success_order_cnt_1d)，同义词=交易成功订单量、trade_success_order_cnt_1d、交易成功订单量-最近1天、订单
- `pay_success_discount_order_cnt_1d`：支付成功优惠单量，角色=METRIC，说明=支付成功优惠单量-最近1天，公式=SUM(pay_success_discount_order_cnt_1d)，同义词=支付成功优惠单量、pay_success_discount_order_cnt_1d、支付成功优惠单量-最近1天、订单
- `pay_success_discount_amt_1d`：支付成功优惠金额元，角色=METRIC，说明=支付成功优惠金额元-最近1天，公式=SUM(pay_success_discount_amt_1d)，同义词=支付成功优惠金额元、pay_success_discount_amt_1d、支付成功优惠金额元-最近1天
- `trade_success_discount_order_cnt_1d`：交易成功优惠单量，角色=METRIC，说明=交易成功优惠单量-最近1天，公式=SUM(trade_success_discount_order_cnt_1d)，同义词=交易成功优惠单量、trade_success_discount_order_cnt_1d、交易成功优惠单量-最近1天、订单
- `trade_success_discount_amt_1d`：交易成功优惠金额元，角色=METRIC，说明=交易成功优惠金额元-最近1天，公式=SUM(trade_success_discount_amt_1d)，同义词=交易成功优惠金额元、trade_success_discount_amt_1d、交易成功优惠金额元-最近1天
- `cs_ticket_cnt_1d`：咨询工单量，角色=METRIC，说明=咨询工单量-最近1天，公式=SUM(cs_ticket_cnt_1d)，同义词=咨询工单量、cs_ticket_cnt_1d、咨询工单量-最近1天
- `risk_complaint_order_cnt_1d`：风控客诉订单量，角色=METRIC，说明=风控客诉订单量-最近1天，公式=SUM(risk_complaint_order_cnt_1d)，同义词=风控客诉订单量、risk_complaint_order_cnt_1d、风控客诉订单量-最近1天、订单
- `seller_repay_order_cnt_1d`：卖家赔付订单数，角色=METRIC，说明=卖家赔付订单数-最近1天，公式=SUM(seller_repay_order_cnt_1d)，同义词=卖家赔付订单数、seller_repay_order_cnt_1d、卖家赔付订单数-最近1天、订单
- `seller_repay_amt_1d`：卖家赔付金额元，角色=METRIC，说明=卖家赔付金额元-最近1天，公式=SUM(seller_repay_amt_1d)，同义词=卖家赔付金额元、seller_repay_amt_1d、卖家赔付金额元-最近1天
- `entry_audit_duration_hours_1d`：入驻审核平均时效小时，角色=METRIC，说明=入驻审核平均时效小时-最近1天，公式=SUM(entry_audit_duration_hours_1d)，同义词=入驻审核平均时效小时、entry_audit_duration_hours_1d、入驻审核平均时效小时-最近1天
- `deposit_pay_cnt_1d`：缴纳保证金次数，角色=METRIC，说明=缴纳保证金次数-最近1天，公式=SUM(deposit_pay_cnt_1d)，同义词=缴纳保证金次数、deposit_pay_cnt_1d、缴纳保证金次数-最近1天
- `entry_success_duration_hours_1d`：入驻成功平均总时长小时，角色=METRIC，说明=入驻成功平均总时长小时-最近1天，公式=SUM(entry_success_duration_hours_1d)，同义词=入驻成功平均总时长小时、entry_success_duration_hours_1d、入驻成功平均总时长小时-最近1天
- `punish_cnt_1d`：处罚次数，角色=METRIC，说明=处罚次数-最近1天；当前按处罚类申诉记录近似，待补处罚单源表校正，公式=SUM(punish_cnt_1d)，同义词=处罚次数、punish_cnt_1d、处罚次数-最近1天；当前按处罚类申诉记录近似，待补处罚单源表校正
- `appeal_cnt_1d`：申诉次数，角色=METRIC，说明=申诉次数-最近1天，公式=SUM(appeal_cnt_1d)，同义词=申诉次数、appeal_cnt_1d、申诉次数-最近1天
- `appeal_success_cnt_1d`：申诉成功次数，角色=METRIC，说明=申诉成功次数-最近1天，公式=SUM(appeal_success_cnt_1d)，同义词=申诉成功次数、appeal_success_cnt_1d、申诉成功次数-最近1天
- `scm_performance_cnt_1d`：供应链履约量，角色=METRIC，说明=供应链履约量-最近1天；按outbound_modify_time大于outbound_latest_time统计，公式=SUM(scm_performance_cnt_1d)，同义词=供应链履约量、scm_performance_cnt_1d、供应链履约量-最近1天；按outbound_modify_time大于outbound_latest_time统计
- `return_success_cnt_1d`：退货成功量，角色=METRIC，说明=退货成功量-最近1天，公式=SUM(return_success_cnt_1d)，同义词=退货成功量、return_success_cnt_1d、退货成功量-最近1天
- `return_cnt_1d`：退货量，角色=METRIC，说明=退货量-最近1天，公式=SUM(return_cnt_1d)，同义词=退货量、return_cnt_1d、退货量-最近1天
- `fake_identify_cnt_1d`：鉴定为假货量，角色=METRIC，说明=鉴定为假货量-最近1天，公式=SUM(fake_identify_cnt_1d)，同义词=鉴定为假货量、fake_identify_cnt_1d、鉴定为假货量-最近1天
- `check_unpass_cnt_1d`：质检不通过量，角色=METRIC，说明=质检不通过量-最近1天，公式=SUM(check_unpass_cnt_1d)，同义词=质检不通过量、check_unpass_cnt_1d、质检不通过量-最近1天
- `goods_audit_reject_cnt_1d`：商品审核拒绝量，角色=METRIC，说明=商品审核拒绝量-最近1天，公式=SUM(goods_audit_reject_cnt_1d)，同义词=商品审核拒绝量、goods_audit_reject_cnt_1d、商品审核拒绝量-最近1天
- `goods_online_cnt_1d`：上架商品量，角色=METRIC，说明=上架商品量-最近1天，公式=SUM(goods_online_cnt_1d)，同义词=上架商品量、goods_online_cnt_1d、上架商品量-最近1天
- `goods_apply_cnt_1d`：商品申请量，角色=METRIC，说明=商品申请量-最近1天，公式=SUM(goods_apply_cnt_1d)，同义词=商品申请量、goods_apply_cnt_1d、商品申请量-最近1天
- `trade_success_gmv_amt_1d`：交易成功gmv金额元，角色=METRIC，说明=交易成功gmv金额元-最近1天，公式=SUM(trade_success_gmv_amt_1d)，同义词=交易成功gmv金额元、trade_success_gmv_amt_1d、交易成功gmv金额元-最近1天、GMV、成交额、销售额、交易流水
- `trade_success_user_cnt_1d`：交易成功用户量，角色=METRIC，说明=交易成功用户量-最近1天，公式=SUM(trade_success_user_cnt_1d)，同义词=交易成功用户量、trade_success_user_cnt_1d、交易成功用户量-最近1天
- `avg_pay_order_amt_1d`：支付成功客单价元，角色=METRIC，说明=支付成功客单价元-最近1天，公式=AVG(avg_pay_order_amt_1d)，同义词=支付成功客单价元、avg_pay_order_amt_1d、支付成功客单价元-最近1天、订单
- `order_close_cnt_1d`：关闭订单量，角色=METRIC，说明=关闭订单量-最近1天，公式=SUM(order_close_cnt_1d)，同义词=关闭订单量、order_close_cnt_1d、关闭订单量-最近1天、订单
- `seller_subsidy_amt_1d`：商家补贴金额元，角色=METRIC，说明=商家补贴金额元-最近1天，公式=SUM(seller_subsidy_amt_1d)，同义词=商家补贴金额元、seller_subsidy_amt_1d、商家补贴金额元-最近1天
- `platform_subsidy_amt_1d`：平台补贴金额元，角色=METRIC，说明=平台补贴金额元-最近1天，公式=SUM(platform_subsidy_amt_1d)，同义词=平台补贴金额元、platform_subsidy_amt_1d、平台补贴金额元-最近1天
- `ship_timeout_order_cnt_1d`：发货超时订单量，角色=METRIC，说明=发货超时订单量-最近1天，公式=SUM(ship_timeout_order_cnt_1d)，同义词=发货超时订单量、ship_timeout_order_cnt_1d、发货超时订单量-最近1天、订单
- `signed_order_cnt_1d`：签收订单量，角色=METRIC，说明=签收订单量-最近1天，公式=SUM(signed_order_cnt_1d)，同义词=签收订单量、signed_order_cnt_1d、签收订单量-最近1天、订单
- `delivery_timeout_order_cnt_1d`：签收超预计送达订单量，角色=METRIC，说明=签收超预计送达订单量-最近1天，公式=SUM(delivery_timeout_order_cnt_1d)，同义词=签收超预计送达订单量、delivery_timeout_order_cnt_1d、签收超预计送达订单量-最近1天、订单
- `pay_discount_rate_1d`：支付成功优惠金额占支付gmv比例，角色=METRIC，说明=支付成功优惠金额占支付gmv比例-最近1天，公式=AVG(pay_discount_rate_1d)，同义词=支付成功优惠金额占支付gmv比例、pay_discount_rate_1d、支付成功优惠金额占支付gmv比例-最近1天
- `refund_amt_1d`：退款金额元，角色=METRIC，说明=退款金额元-最近1天，公式=SUM(refund_amt_1d)，同义词=退款金额元、refund_amt_1d、退款金额元-最近1天、退款、售后
- `return_success_amt_1d`：退货成功金额元，角色=METRIC，说明=退货成功金额元-最近1天，公式=SUM(return_success_amt_1d)，同义词=退货成功金额元、return_success_amt_1d、退货成功金额元-最近1天
- `seller_responsible_refund_cnt_1d`：卖家责任退货量，角色=METRIC，说明=卖家责任退货量-最近1天，公式=SUM(seller_responsible_refund_cnt_1d)，同义词=卖家责任退货量、seller_responsible_refund_cnt_1d、卖家责任退货量-最近1天、退款、售后
- `direct_refund_cnt_1d`：直接退款量，角色=METRIC，说明=直接退款量-最近1天，公式=SUM(direct_refund_cnt_1d)，同义词=直接退款量、direct_refund_cnt_1d、直接退款量-最近1天
- `refund_rate_1d`：退货量占支付订单量比例，角色=METRIC，说明=退货量占支付订单量比例-最近1天，公式=AVG(refund_rate_1d)，同义词=退货量占支付订单量比例、退款率、退货率、售后率、整体退款率、店铺退款率、refund_rate、refund_rate_1d、退货量占支付订单量比例-最近1天、退款、售后、订单
- `ticket_reopen_cnt_1d`：工单二次开启量，角色=METRIC，说明=工单二次开启量-最近1天，公式=SUM(ticket_reopen_cnt_1d)，同义词=工单二次开启量、ticket_reopen_cnt_1d、工单二次开启量-最近1天
- `ticket_reminder_cnt_1d`：催单工单量，角色=METRIC，说明=催单工单量-最近1天，公式=SUM(ticket_reminder_cnt_1d)，同义词=催单工单量、ticket_reminder_cnt_1d、催单工单量-最近1天
- `ticket_close_cnt_1d`：关闭工单量，角色=METRIC，说明=关闭工单量-最近1天，公式=SUM(ticket_close_cnt_1d)，同义词=关闭工单量、ticket_close_cnt_1d、关闭工单量-最近1天
- `avg_ticket_score_1d`：平均工单评价分，角色=METRIC，说明=平均工单评价分-最近1天，公式=AVG(avg_ticket_score_1d)，同义词=平均工单评价分、avg_ticket_score_1d、平均工单评价分-最近1天
- `goods_audit_pass_cnt_1d`：商品审核通过量，角色=METRIC，说明=商品审核通过量-最近1天，公式=SUM(goods_audit_pass_cnt_1d)，同义词=商品审核通过量、goods_audit_pass_cnt_1d、商品审核通过量-最近1天
