# 客服理赔 / dwm_cs_repay_detail_df

状态：PENDING_REVIEW
表说明：dwm-客服域-理赔全流程表-全量表
数据粒度：订单/子订单明细粒度
时间字段：`pt`
商家过滤字段：`seller_id`
人工业务说明：客服理赔全流程表，承载赔付单、赔付金额、审批状态、到账状态与赔付原因

## 列级语义
- `bill_id`：赔付单号，角色=KEY，说明=赔付单号，PF190827160812345，同义词=赔付单号、bill_id、赔付单号，PF190827160812345
- `pt`：业务日期，角色=TIME，说明=业务日期，同义词=业务日期、pt
- `cause_id`：赔付原因，角色=KEY，说明=赔付原因，同义词=赔付原因、cause_id
- `order_id`：订单号，角色=KEY，说明=订单号，同义词=订单号、order_id、订单
- `is_return`：是否已退货 0否 1是，角色=DIMENSION，说明=是否已退货 0否 1是，同义词=是否已退货 0否 1是、is_return
- `repay_amt`：支付金额，角色=METRIC，说明=支付金额，公式=SUM(repay_amt)，同义词=支付金额、repay_amt
- `ticket_id`：工单号，角色=KEY，说明=工单号，同义词=工单号、ticket_id
- `buyer_id`：赔付买家id，角色=KEY，说明=赔付买家id，同义词=赔付买家id、buyer_id
- `buyer_name`：赔付买家用户名，角色=DIMENSION，说明=赔付买家用户名，同义词=赔付买家用户名、buyer_name
- `seller_id`：赔付卖家用户id，角色=KEY，说明=赔付卖家用户id，同义词=赔付卖家用户id、seller_id
- `seller_name`：赔付卖家用户名，角色=DIMENSION，说明=赔付卖家用户名，同义词=赔付卖家用户名、seller_name
- `content`：备注，角色=OTHER，说明=备注，同义词=备注、content
- `reject_content`：驳回原因，角色=DIMENSION，说明=驳回原因，同义词=驳回原因、reject_content
- `repay_status_code`：状态code，角色=DIMENSION，说明=状态code，1审批中，2已驳回，3审批完成, 4已取消，同义词=状态code、repay_status_code、状态code，1审批中，2已驳回，3审批完成, 4已取消
- `repay_status_name`：状态名称，角色=DIMENSION，说明=状态名称，同义词=状态名称、repay_status_name
- `creator_id`：建立人员，角色=KEY，说明=建立人员，同义词=建立人员、creator_id
- `create_time`：系统时间，角色=TIME，说明=系统时间，同义词=系统时间、create_time
- `modifier_id`：修改人员，角色=KEY，说明=修改人员，同义词=修改人员、modifier_id
- `modify_time`：修改时间，角色=TIME，说明=修改时间，同义词=修改时间、modify_time
- `pay_status_code`：到账状态code 1打款中 2打款失败 3打款成功，角色=DIMENSION，说明=到账状态code 1打款中 2打款失败 3打款成功，同义词=到账状态code 1打款中 2打款失败 3打款成功、pay_status_code
- `pay_status_name`：到账状态name，角色=DIMENSION，说明=到账状态name，同义词=到账状态name、pay_status_name
- `pay_way_code`：赔款方式code 1优惠券2现金3语兴好物币，角色=DIMENSION，说明=赔款方式code 1优惠券2现金3语兴好物币，同义词=赔款方式code 1优惠券2现金3语兴好物币、pay_way_code
- `pay_way_name`：赔款方式name，角色=DIMENSION，说明=赔款方式name，同义词=赔款方式name、pay_way_name
- `sub_order_id`：子订单号，角色=KEY，说明=子订单号，同义词=子订单号、sub_order_id、订单
- `express_id`：运单号，角色=KEY，说明=运单号，同义词=运单号、express_id
- `pay_account`：支付账户，角色=OTHER，说明=支付账户，同义词=支付账户、pay_account
- `process_id`：流程ID，角色=KEY，说明=流程ID，同义词=流程ID、process_id
- `reason_code`：赔付原因，角色=DIMENSION，说明=赔付原因，同义词=赔付原因、reason_code
- `coupon_type_code`：优惠券类型code 1满减券 2包邮券，角色=DIMENSION，说明=优惠券类型code 1满减券 2包邮券，同义词=优惠券类型code 1满减券 2包邮券、coupon_type_code
- `coupon_type_name`：优惠券类型name，角色=DIMENSION，说明=优惠券类型name，同义词=优惠券类型name、coupon_type_name
- `coupon_rule_a`：满多少元，角色=METRIC，说明=满多少元，公式=SUM(coupon_rule_a)，同义词=满多少元、coupon_rule_a
- `coupon_rule_b`：减多少元，角色=METRIC，说明=减多少元，公式=SUM(coupon_rule_b)，同义词=减多少元、coupon_rule_b
- `coupon_time_limit`：优惠券使用期限，角色=OTHER，说明=优惠券使用期限，格式：2019-08-25,2019-08-28，同义词=优惠券使用期限、coupon_time_limit、优惠券使用期限，格式：2019-08-25,2019-08-28
- `coupon_id`：优惠券id，角色=KEY，说明=优惠券id，同义词=优惠券id、coupon_id
- `activity_id`：活动id，角色=KEY，说明=活动id，同义词=活动id、activity_id
- `level1_reason_code`：一级理赔理由code，角色=DIMENSION，说明=一级理赔理由code，同义词=一级理赔理由code、level1_reason_code
- `level1_reason_name`：一级理赔理由name，角色=DIMENSION，说明=一级理赔理由name，同义词=一级理赔理由name、level1_reason_name
- `level2_reason_code`：二级理赔理由id，角色=DIMENSION，说明=二级理赔理由id，同义词=二级理赔理由id、level2_reason_code
- `level2_reason_name`：二级理赔理由name，角色=DIMENSION，说明=二级理赔理由name，同义词=二级理赔理由name、level2_reason_name
- `level3_reason_code`：三级理赔理由id，角色=DIMENSION，说明=三级理赔理由id，同义词=三级理赔理由id、level3_reason_code
- `level3_reason_name`：三级理赔理由name，角色=DIMENSION，说明=三级理赔理由name，同义词=三级理赔理由name、level3_reason_name
