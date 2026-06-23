# 电商优惠券 / dwm_coupon_detail_di

状态：PENDING_REVIEW
表说明：dwm-优惠券域-明细表-增量表
数据粒度：优惠券/津贴明细粒度
时间字段：`pt`
商家过滤字段：`seller_id`
人工业务说明：优惠券/津贴明细表，承载优惠券发放、领取、退回、优惠金额与津贴金额

## 列级语义
- `coupon_id`：券编号，角色=KEY，说明=券编号，同义词=券编号、coupon_id、优惠、优惠券、券
- `user_id`：用户id，角色=KEY，说明=用户id，同义词=用户id、user_id
- `suit_owner_code`：适用方code，角色=DIMENSION，说明=适用方code，同义词=适用方code、suit_owner_code
- `suit_owner_name`：适用方name，角色=DIMENSION，说明=适用方name，同义词=适用方name、suit_owner_name
- `template_id`：优惠券模板编号，角色=KEY，说明=优惠券模板编号，同义词=优惠券模板编号、template_id、优惠、优惠券、券
- `template_title`：标题，角色=OTHER，说明=标题，同义词=标题、template_title
- `activity_id`：活动id，角色=KEY，说明=活动id，同义词=活动id、activity_id
- `coupon_amt`：优惠金额，角色=METRIC，说明=优惠金额，公式=SUM(coupon_amt)，同义词=优惠金额、coupon_amt、优惠、优惠券、券
- `discount_way_code`：优惠方式code，角色=DIMENSION，说明=优惠方式code，同义词=优惠方式code、discount_way_code、优惠、优惠券、券
- `discount_way_name`：优惠方式name，角色=DIMENSION，说明=优惠方式name，同义词=优惠方式name、discount_way_name、优惠、优惠券、券
- `found_type_code`：款项code，角色=DIMENSION，说明=款项code，同义词=款项code、found_type_code
- `found_type_name`：款项name，角色=DIMENSION，说明=款项name，同义词=款项name、found_type_name
- `coupon_start_time`：开始时间，角色=TIME，说明=开始时间，同义词=开始时间、coupon_start_time、优惠、优惠券、券
- `coupon_expire_time`：过期时间，角色=TIME，说明=过期时间，同义词=过期时间、coupon_expire_time、优惠、优惠券、券
- `coupon_send_status_code`：状态code，角色=DIMENSION，说明=状态code，同义词=状态code、coupon_send_status_code、优惠、优惠券、券
- `coupon_send_status_name`：状态name，角色=DIMENSION，说明=状态name，同义词=状态name、coupon_send_status_name、优惠、优惠券、券
- `threshold`：门槛，角色=OTHER，说明=门槛，同义词=门槛、threshold
- `send_source_name`：推送来源，角色=DIMENSION，说明=推送来源，同义词=推送来源、send_source_name
- `coupon_content`：优惠券信息，角色=OTHER，说明=优惠券信息，同义词=优惠券信息、coupon_content、优惠、优惠券、券
- `coupon_create_time`：优惠券创建时间，角色=TIME，说明=优惠券创建时间，同义词=优惠券创建时间、coupon_create_time、优惠、优惠券、券
- `coupon_modify_time`：优惠券更新时间，角色=TIME，说明=优惠券更新时间，同义词=优惠券更新时间、coupon_modify_time、优惠、优惠券、券
- `is_receive`：是否抢到优惠券，角色=DIMENSION，说明=是否抢到优惠券，同义词=是否抢到优惠券、is_receive、优惠、优惠券、券
- `snap_create_time`：抢券创建时间，角色=TIME，说明=抢券创建时间，同义词=抢券创建时间、snap_create_time
- `snap_modify_time`：抢券变更时间，角色=TIME，说明=抢券变更时间，同义词=抢券变更时间、snap_modify_time
- `coupon_refund_id`：优惠券退回编号，角色=KEY，说明=优惠券退回编号，同义词=优惠券退回编号、coupon_refund_id、退款、售后、优惠、优惠券、券
- `refund_time`：优惠券退回时间，角色=TIME，说明=优惠券退回时间，同义词=优惠券退回时间、refund_time、退款、售后、优惠、优惠券、券
- `allowance_amt`：津贴金额，角色=METRIC，说明=津贴金额，公式=SUM(CAST(allowance_amt AS DECIMAL(18,2)))，同义词=津贴金额、allowance_amt
- `seller_id`：津贴商家id，角色=KEY，说明=津贴商家id，同义词=津贴商家id、seller_id
- `seller_name`：津贴商家name，角色=DIMENSION，说明=津贴商家name，同义词=津贴商家name、seller_name
- `allowance_create_time`：创建时间，角色=TIME，说明=创建时间，同义词=创建时间、allowance_create_time
- `allowance_modify_time`：变更时间，角色=TIME，说明=变更时间，同义词=变更时间、allowance_modify_time
- `template_createtor_id`：优惠券创建人id，角色=KEY，说明=优惠券创建人id，同义词=优惠券创建人id、template_createtor_id、优惠、优惠券、券
- `template_createtor_name`：优惠券创建人name，角色=DIMENSION，说明=优惠券创建人name，同义词=优惠券创建人name、template_createtor_name、优惠、优惠券、券
- `template_remark`：优惠券备注信息，角色=OTHER，说明=优惠券备注信息，同义词=优惠券备注信息、template_remark、优惠、优惠券、券
- `is_voucher`：是否代金券，角色=DIMENSION，说明=是否代金券，同义词=是否代金券、is_voucher
- `template_create_time`：优惠券创建时间，角色=TIME，说明=优惠券创建时间，同义词=优惠券创建时间、template_create_time、优惠、优惠券、券
- `template_modify_time`：变更时间，角色=TIME，说明=变更时间，同义词=变更时间、template_modify_time
- `template_valid_dur`：模板有效期间，角色=OTHER，说明=模板有效期间，天，同义词=模板有效期间、template_valid_dur、模板有效期间，天
- `activity_name`：优惠券活动name，角色=DIMENSION，说明=优惠券活动name，同义词=优惠券活动name、activity_name、优惠、优惠券、券
- `activity_start_time`：优惠券活动开始时间，角色=TIME，说明=优惠券活动开始时间，同义词=优惠券活动开始时间、activity_start_time、优惠、优惠券、券
- `activity_expire_time`：优惠券活动结束时间，角色=TIME，说明=优惠券活动结束时间，同义词=优惠券活动结束时间、activity_expire_time、优惠、优惠券、券
- `activity_createtor_id`：优惠券活动创建人id，角色=KEY，说明=优惠券活动创建人id，同义词=优惠券活动创建人id、activity_createtor_id、优惠、优惠券、券
- `activity_createtor_name`：优惠券活动创建人name，角色=DIMENSION，说明=优惠券活动创建人name，同义词=优惠券活动创建人name、activity_createtor_name、优惠、优惠券、券
- `activity_create_time`：优惠券活动创建时间，角色=TIME，说明=优惠券活动创建时间，同义词=优惠券活动创建时间、activity_create_time、优惠、优惠券、券
- `activity_modify_time`：优惠券活动变更时间，角色=TIME，说明=优惠券活动变更时间，同义词=优惠券活动变更时间、activity_modify_time、优惠、优惠券、券
- `pt`：业务日期，角色=TIME，说明=业务日期，同义词=业务日期、pt
