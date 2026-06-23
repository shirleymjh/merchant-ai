# 身份信息 / dim_merchant_df

状态：PENDING_REVIEW
表说明：dim-商家域-商家信息表
数据粒度：商家主体/资质快照粒度
时间字段：`pt`
商家过滤字段：`merchant_id`
人工业务说明：商家信息维表，承载商家主体、资质、入驻状态、结算、发货模式和保证金余额

## 列级语义
- `user_id`：用户id，角色=KEY，说明=用户id，同义词=用户id、user_id
- `merchant_id`：商家id，角色=KEY，说明=商家id，同义词=商家id、merchant_id
- `merchant_type_code`：商户类型code 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6，角色=DIMENSION，说明=商户类型code 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方，同义词=商户类型code 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6、merchant_type_code、商户类型code 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方
- `merchant_type_name`：商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6，角色=DIMENSION，说明=商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方，同义词=商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6、merchant_type_name、商户类型name 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家 5.个人普通商户 6-品牌方
- `brand_type_code`：资质类型code，角色=DIMENSION，说明=资质类型code -1代表商家是服务商 0、商家入驻时填基本信息，还没填资质信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商，同义词=资质类型code、brand_type_code、资质类型code -1代表商家是服务商 0、商家入驻时填基本信息，还没填资质信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商
- `brand_type_name`：资质类型name，角色=DIMENSION，说明=资质类型name -1代表商家是服务商 0、商家入驻时填基本信息，还没填资质信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商，同义词=资质类型name、brand_type_name、资质类型name -1代表商家是服务商 0、商家入驻时填基本信息，还没填资质信息 1.品牌方 2.经销商 3.市场贸易商 4.扫货商 5.零售百货商
- `audit_user_list`：审批人aclIds，角色=OTHER，说明=审批人aclIds，逗号分隔，同义词=审批人aclIds、audit_user_list、审批人aclIds，逗号分隔
- `merchant_apply_status_code`：审核状态code，角色=DIMENSION，说明=审核状态code，商家提交资料-0，保证金/账单/费率/协议填写审核-100，保证金低于标准，二次审核-110，初审不通过-120，初审通过，等待特批费率/类目运营审核-130，初审完成-140，待法务审核-200，法务审核不通过-210，法务审核通过，商家协议待确认-300，商家协议确认成功，保证金待缴纳-500，保证金转账待审核-520，保证金转账审核不通过-530，商家入驻被驳回-600，保证金转账审核通过，入驻完成-800，同义词=审核状态code、merchant_apply_status_code、审核状态code，商家提交资料-0，保证金/账单/费率/协议填写审核-100，保证金低于标准，二次审核-110，初审不通过-120，初审通过，等待特批费率/类目运营审核-130，初审完成-140，待法务审核-200，法务审核不通过-210，法务审核通过，商家协议待确认-300，商家协议确认成功，保证金待缴纳-500，保证金转账待审核-520，保证金转账审核不通过-530，商家入驻被驳回-600，保证金转账审核通过，入驻完成-800、保证金、押金
- `merchant_apply_status_name`：审核状态name，角色=DIMENSION，说明=审核状态name，同义词=审核状态name、merchant_apply_status_name
- `init_deposit_amt`：商户入驻初始保证金元，角色=METRIC，说明=商户入驻初始保证金元，公式=SUM(init_deposit_amt)，同义词=商户入驻初始保证金元、init_deposit_amt、保证金、押金
- `balance_type_code`：结算类型code 0.实时 1.周结 2.半月结 3.月结，角色=DIMENSION，说明=结算类型code 0.实时 1.周结 2.半月结 3.月结，同义词=结算类型code 0.实时 1.周结 2.半月结 3.月结、balance_type_code
- `balance_type_name`：结算类型name 0.实时 1.周结 2.半月结 3.月结，角色=DIMENSION，说明=结算类型name 0.实时 1.周结 2.半月结 3.月结，同义词=结算类型name 0.实时 1.周结 2.半月结 3.月结、balance_type_name
- `mobile`：商家手机号，角色=OTHER，说明=商家手机号，同义词=商家手机号、mobile
- `company_name`：公司名称，角色=DIMENSION，说明=公司名称，同义词=公司名称、company_name
- `license_id`：营业执照编号，角色=KEY，说明=营业执照编号，同义词=营业执照编号、license_id
- `license_period_start`：营业执照有效期 开始时间，角色=TIME，说明=营业执照有效期 开始时间，同义词=营业执照有效期 开始时间、license_period_start
- `license_period_end`：营业执照有效期 结束时间，角色=TIME，说明=营业执照有效期 结束时间，同义词=营业执照有效期 结束时间、license_period_end
- `company_province`：省，角色=DIMENSION，说明=省，同义词=省、company_province
- `company_city`：市，角色=DIMENSION，说明=市，同义词=市、company_city
- `company_district`：区，角色=OTHER，说明=区，同义词=区、company_district
- `company_address`：详细地址，角色=OTHER，说明=详细地址，同义词=详细地址、company_address
- `corporation_name`：法人姓名，角色=DIMENSION，说明=法人姓名，同义词=法人姓名、corporation_name
- `corporation_idcard`：法人身份证号，角色=OTHER，说明=法人身份证号，同义词=法人身份证号、corporation_idcard
- `corporation_idcard_period_start`：法人身份有效期 开始时间，角色=TIME，说明=法人身份有效期 开始时间，同义词=法人身份有效期 开始时间、corporation_idcard_period_start
- `corporation_idcard_period_end`：法人身份有效期 结束时间，角色=TIME，说明=法人身份有效期 结束时间，同义词=法人身份有效期 结束时间、corporation_idcard_period_end
- `is_unconditional_refund`：是否支持七天无理由退货 0 否 1 是，角色=DIMENSION，说明=是否支持七天无理由退货 0 否 1 是，同义词=是否支持七天无理由退货 0 否 1 是、is_unconditional_refund、退款、售后
- `is_invoice`：是否支持开具发票 0否 1是，角色=DIMENSION，说明=是否支持开具发票 0否 1是，同义词=是否支持开具发票 0否 1是、is_invoice
- `refund_mobile`：退货地址收货人手机号，角色=OTHER，说明=退货地址收货人手机号，同义词=退货地址收货人手机号、refund_mobile、退款、售后
- `inviter`：入驻邀约人姓名，角色=OTHER，说明=入驻邀约人姓名，同义词=入驻邀约人姓名、inviter
- `inviter_mobile`：入驻邀约人手机号，角色=OTHER，说明=入驻邀约人手机号，同义词=入驻邀约人手机号、inviter_mobile
- `currency`：币种，角色=OTHER，说明=币种，同义词=币种、currency
- `ship_model_code`：发货模式code 1，角色=DIMENSION，说明=发货模式code 1-集货模式 2-自发货模式，同义词=发货模式code 1、ship_model_code、发货模式code 1-集货模式 2-自发货模式
- `ship_model_name`：发货模式name 1，角色=DIMENSION，说明=发货模式name 1-集货模式 2-自发货模式，同义词=发货模式name 1、ship_model_name、发货模式name 1-集货模式 2-自发货模式
- `contact_name`：联系人姓名，角色=DIMENSION，说明=联系人姓名，同义词=联系人姓名、contact_name
- `contact_idcard`：联系人身份证号，角色=OTHER，说明=联系人身份证号，同义词=联系人身份证号、contact_idcard
- `contact_idcard_period_start`：联系人身份有效期 开始时间，角色=TIME，说明=联系人身份有效期 开始时间，同义词=联系人身份有效期 开始时间、contact_idcard_period_start
- `contact_idcard_period_end`：联系人身份有效期 结束时间，角色=TIME，说明=联系人身份有效期 结束时间，同义词=联系人身份有效期 结束时间、contact_idcard_period_end
- `contact_idcard_type`：联系人默认大陆身份证，角色=DIMENSION，说明=联系人默认大陆身份证，类型包括：100 大陆身份证；105 港澳居民往来内地通行证；106 台湾同胞往来大陆通行证；108 外国人居留证，同义词=联系人默认大陆身份证、contact_idcard_type、联系人默认大陆身份证，类型包括：100 大陆身份证；105 港澳居民往来内地通行证；106 台湾同胞往来大陆通行证；108 外国人居留证
- `contact_mobile`：联系人手机号，角色=OTHER，说明=联系人手机号，同义词=联系人手机号、contact_mobile
- `business_province`：经营商省地区，角色=DIMENSION，说明=经营商省地区，同义词=经营商省地区、business_province
- `business_city`：经营商所在城市地区，角色=DIMENSION，说明=经营商所在城市地区，同义词=经营商所在城市地区、business_city
- `business_district`：经营商街道地区，角色=OTHER，说明=经营商街道地区，同义词=经营商街道地区、business_district
- `business_address`：经营商详细地址，角色=OTHER，说明=经营商详细地址，同义词=经营商详细地址、business_address
- `send_province`：发货省，角色=DIMENSION，说明=发货省，同义词=发货省、send_province
- `send_city`：发货市，角色=DIMENSION，说明=发货市，同义词=发货市、send_city
- `send_district`：发货区，角色=OTHER，说明=发货区，同义词=发货区、send_district
- `send_street`：退货地址街道，角色=OTHER，说明=退货地址街道，同义词=退货地址街道、send_street
- `send_address`：发货详细地址，角色=OTHER，说明=发货详细地址，同义词=发货详细地址、send_address
- `refnd_province`：退货地址省，角色=DIMENSION，说明=退货地址省，同义词=退货地址省、refnd_province
- `refnd_city`：退货地址市，角色=DIMENSION，说明=退货地址市，同义词=退货地址市、refnd_city
- `refnd_district`：退货地址区，角色=OTHER，说明=退货地址区，同义词=退货地址区、refnd_district
- `refnd_street`：退货地址街道，角色=OTHER，说明=退货地址街道，同义词=退货地址街道、refnd_street
- `refnd_address`：退货地址详细地址，角色=OTHER，说明=退货地址详细地址，同义词=退货地址详细地址、refnd_address
- `service_version`：协议版本，角色=OTHER，说明=协议版本，同义词=协议版本、service_version
- `min_poundage`：手续费下限，角色=OTHER，说明=手续费下限，同义词=手续费下限、min_poundage
- `max_poundage`：手续费上限，角色=OTHER，说明=手续费上限，同义词=手续费上限、max_poundage
- `bank_name`：开户行，角色=DIMENSION，说明=开户行，同义词=开户行、bank_name
- `bank_account`：银行账号，角色=OTHER，说明=银行账号，同义词=银行账号、bank_account
- `account_type_code`：账户类型code 1.对公 2.对私，角色=DIMENSION，说明=账户类型code 1.对公 2.对私，同义词=账户类型code 1.对公 2.对私、account_type_code
- `account_type_name`：账户类型name 1.对公 2.对私，角色=DIMENSION，说明=账户类型name 1.对公 2.对私，同义词=账户类型name 1.对公 2.对私、account_type_name
- `deposit_freeze`：冻结保证金，角色=OTHER，说明=冻结保证金，同义词=冻结保证金、deposit_freeze、保证金、押金
- `deposit_amt`：保证金，角色=METRIC，说明=保证金，公式=SUM(deposit_amt)，同义词=保证金、deposit_amt、押金
- `poundage_discount`：费率折扣，角色=OTHER，说明=费率折扣，同义词=费率折扣、poundage_discount
- `discount_start_time`：费率折扣开始时间，角色=TIME，说明=费率折扣开始时间，同义词=费率折扣开始时间、discount_start_time
- `discount_end_time`：费率折扣结束时间，角色=TIME，说明=费率折扣结束时间，同义词=费率折扣结束时间、discount_end_time
- `create_time`：添加时间，角色=TIME，说明=添加时间，同义词=添加时间、create_time
- `modify_time`：修改时间，角色=TIME，说明=修改时间，同义词=修改时间、modify_time
- `pt`：日期分区yyyyMMdd，角色=TIME，说明=日期分区yyyyMMdd，同义词=日期分区yyyyMMdd、pt
