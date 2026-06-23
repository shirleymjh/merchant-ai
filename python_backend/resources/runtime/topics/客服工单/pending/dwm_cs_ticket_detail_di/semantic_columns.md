# 客服工单 / dwm_cs_ticket_detail_di

状态：PENDING_REVIEW
表说明：dwm-客服域-工单域-明细宽表-增量表
数据粒度：订单/子订单明细粒度
时间字段：`pt`
商家过滤字段：`seller_id`
人工业务说明：客服工单明细表，承载咨询、催单、关闭、评分、状态等服务过程

## 列级语义
- `ticket_id`：工单编号，角色=KEY，说明=工单编号，同义词=工单编号、ticket_id
- `pt`：业务日期，角色=TIME，说明=业务日期，同义词=业务日期、pt
- `ticket_title`：工单标题，角色=OTHER，说明=工单标题，同义词=工单标题、ticket_title
- `session_id`：会话id，角色=KEY，说明=会话id，同义词=会话id、session_id
- `call_id`：通话id，角色=KEY，说明=通话id，同义词=通话id、call_id
- `ticket_source_code`：末级来源id，角色=DIMENSION，说明=末级来源id，同义词=末级来源id、ticket_source_code
- `priority_code`：工单等级类型code，角色=DIMENSION，说明=工单等级类型code，同义词=工单等级类型code、priority_code
- `priority_name`：工单等级类型name，角色=DIMENSION，说明=工单等级类型name，普通、加急，同义词=工单等级类型name、priority_name、工单等级类型name，普通、加急
- `ticket_type_code`：工单类型code，角色=DIMENSION，说明=工单类型code，存储最后级别的分类id，同义词=工单类型code、ticket_type_code、工单类型code，存储最后级别的分类id
- `user_type_code`：客户身份，角色=DIMENSION，说明=客户身份，0买家、1卖家，同义词=客户身份、user_type_code、客户身份，0买家、1卖家
- `user_type_name`：客户身份，角色=DIMENSION，说明=客户身份，买家、卖家，同义词=客户身份、user_type_name、客户身份，买家、卖家
- `start_operator_id`：工单发起人，角色=KEY，说明=工单发起人，同义词=工单发起人、start_operator_id
- `assignee_operator_id`：工单跟进人，角色=KEY，说明=工单跟进人，同义词=工单跟进人、assignee_operator_id
- `assignee_group_id`：处理组id，角色=KEY，说明=处理组id，同义词=处理组id、assignee_group_id
- `is_reopen`：是否二次开启 0否 1时，角色=DIMENSION，说明=是否二次开启 0否 1时，同义词=是否二次开启 0否 1时、is_reopen
- `is_forward`：是否流转 0 否 1 是，角色=DIMENSION，说明=是否流转 0 否 1 是，同义词=是否流转 0 否 1 是、is_forward
- `responsible_type_code`：责任方code，角色=DIMENSION，说明=责任方code，1买家 2卖家 3客户，同义词=责任方code、responsible_type_code、责任方code，1买家 2卖家 3客户
- `responsible_type_name`：责任方name，角色=DIMENSION，说明=责任方name，买家 卖家 客户，同义词=责任方name、responsible_type_name、责任方name，买家 卖家 客户
- `close_time`：工单完成时间，角色=TIME，说明=工单完成时间，同义词=工单完成时间、close_time
- `follow_time`：工单相应时间，角色=TIME，说明=工单相应时间，第一次消息回复消息，同义词=工单相应时间、follow_time、工单相应时间，第一次消息回复消息
- `chat_time`：工单会话，角色=TIME，说明=工单会话，同义词=工单会话、chat_time
- `evaluate_time`：工单评价会话，角色=TIME，说明=工单评价会话，同义词=工单评价会话、evaluate_time
- `cancel_time`：工单取消时间，角色=TIME，说明=工单取消时间，同义词=工单取消时间、cancel_time
- `ticket_status_code`：工单状态code，角色=DIMENSION，说明=工单状态code，同义词=工单状态code、ticket_status_code
- `ticket_status_name`：工单状态name，角色=DIMENSION，说明=工单状态name，同义词=工单状态name、ticket_status_name
- `spu_id`：商品id，角色=KEY，说明=商品id，同义词=商品id、spu_id
- `is_reminder`：是否催单 0 否 1 是，角色=DIMENSION，说明=是否催单 0 否 1 是，同义词=是否催单 0 否 1 是、is_reminder
- `reminder_cnt`：催单次数，角色=METRIC，说明=催单次数，公式=SUM(reminder_cnt)，同义词=催单次数、reminder_cnt
- `evaluation_status_code`：评价状态code，角色=DIMENSION，说明=评价状态code，0未邀评 1买家邀评 2卖家已邀评 3 卖家买家已邀评，同义词=评价状态code、evaluation_status_code、评价状态code，0未邀评 1买家邀评 2卖家已邀评 3 卖家买家已邀评
- `evaluation_status_name`：评价状态name，角色=DIMENSION，说明=评价状态name，同义词=评价状态name、evaluation_status_name
- `ticket_score`：工单评价分数，角色=METRIC，说明=工单评价分数，公式=AVG(ticket_score)，同义词=工单评价分数、ticket_score
- `ticket_create_time`：工单创建时间，角色=TIME，说明=工单创建时间，同义词=工单创建时间、ticket_create_time
- `ticket_modify_time`：工单更新时间，角色=TIME，说明=工单更新时间，同义词=工单更新时间、ticket_modify_time
- `order_id`：订单号，角色=KEY，说明=订单号，同义词=订单号、order_id、订单
- `sub_order_id`：子订单号，角色=KEY，说明=子订单号，同义词=子订单号、sub_order_id、订单
- `buyer_id`：买家id，角色=KEY，说明=买家id，同义词=买家id、buyer_id
- `buyer_name`：买家用户昵称，角色=DIMENSION，说明=买家用户昵称，同义词=买家用户昵称、buyer_name
- `buyer_real_name`：买家用户姓名，角色=DIMENSION，说明=买家用户姓名，同义词=买家用户姓名、buyer_real_name
- `buyer_mobile`：买家用户手机，角色=OTHER，说明=买家用户手机，同义词=买家用户手机、buyer_mobile
- `seller_type_code`：商户类型code 0.个人vip商户 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家  5.个人普通商户，角色=DIMENSION，说明=商户类型code 0.个人vip商户 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家  5.个人普通商户，同义词=商户类型code 0.个人vip商户 1.急速发货商户 2.企业商户 3.认证商户 4.跨境商家  5.个人普通商户、seller_type_code
- `seller_type_name`：商户类型name，角色=DIMENSION，说明=商户类型name，同义词=商户类型name、seller_type_name
- `seller_express_id`：卖家运单号，角色=KEY，说明=卖家运单号，指卖家发往平台的运单，同义词=卖家运单号、seller_express_id、卖家运单号，指卖家发往平台的运单
- `seller_express_type_code`：卖家物流类型code，角色=DIMENSION，说明=卖家物流类型code，同义词=卖家物流类型code、seller_express_type_code
- `seller_express_type_name`：卖家物流类型name，角色=DIMENSION，说明=卖家物流类型name，同义词=卖家物流类型name、seller_express_type_name
- `buyer_express_id`：买家运单号，角色=KEY，说明=买家运单号，指平台发往用户的运单，同义词=买家运单号、buyer_express_id、买家运单号，指平台发往用户的运单
- `buyer_express_type_code`：买家物流类型code，角色=DIMENSION，说明=买家物流类型code，同义词=买家物流类型code、buyer_express_type_code
- `buyer_express_type_name`：买家物流类型name，角色=DIMENSION，说明=买家物流类型name，同义词=买家物流类型name、buyer_express_type_name
- `seller_id`：卖家用户id，角色=KEY，说明=卖家用户id，同义词=卖家用户id、seller_id
- `seller_name`：卖家用户昵称，角色=DIMENSION，说明=卖家用户昵称，同义词=卖家用户昵称、seller_name
- `seller_real_name`：卖家用户姓名，角色=DIMENSION，说明=卖家用户姓名，同义词=卖家用户姓名、seller_real_name
- `seller_mobile`：卖家联系方式，角色=DIMENSION，说明=卖家联系方式，同义词=卖家联系方式、seller_mobile
- `ticket_content`：留言信息，角色=OTHER，说明=留言信息，富文本，同义词=留言信息、ticket_content、留言信息，富文本
- `msg_content_list`：对话list，角色=OTHER，说明=对话list，同义词=对话list、msg_content_list
- `attachment_url_list`：附件list，角色=OTHER，说明=附件list，同义词=附件list、attachment_url_list
- `check_result`：质检结果，角色=OTHER，说明=质检结果,通过、未通过，同义词=质检结果、check_result、质检结果,通过、未通过
- `check_remark`：质检备注，角色=OTHER，说明=质检备注，同义词=质检备注、check_remark
- `level1_source_code`：一级来源code，角色=DIMENSION，说明=一级来源code，同义词=一级来源code、level1_source_code
- `level1_source_name`：一级来源名称，角色=DIMENSION，说明=一级来源名称，同义词=一级来源名称、level1_source_name
- `level2_source_code`：二级来源code，角色=DIMENSION，说明=二级来源code，同义词=二级来源code、level2_source_code
- `level2_source_name`：二级来源名称，角色=DIMENSION，说明=二级来源名称，同义词=二级来源名称、level2_source_name
- `level3_source_code`：三级来源code，角色=DIMENSION，说明=三级来源code，同义词=三级来源code、level3_source_code
- `level3_source_name`：三级来源名称，角色=DIMENSION，说明=三级来源名称，同义词=三级来源名称、level3_source_name
- `level1_type_code`：工单一级类型id，角色=DIMENSION，说明=工单一级类型id，同义词=工单一级类型id、level1_type_code
- `level1_type_name`：工单一级类型，角色=DIMENSION，说明=工单一级类型，同义词=工单一级类型、level1_type_name
- `level2_type_code`：工单二级类型id，角色=DIMENSION，说明=工单二级类型id，同义词=工单二级类型id、level2_type_code
- `level2_type_name`：工单二级类型，角色=DIMENSION，说明=工单二级类型，同义词=工单二级类型、level2_type_name
- `level3_type_code`：工单三级类型id，角色=DIMENSION，说明=工单三级类型id，同义词=工单三级类型id、level3_type_code
- `level3_type_name`：工单三级类型，角色=DIMENSION，说明=工单三级类型，同义词=工单三级类型、level3_type_name
- `operator_name`：工单操作人用户名，角色=DIMENSION，说明=工单操作人用户名，同义词=工单操作人用户名、operator_name
- `real_name`：真实姓名，角色=DIMENSION，说明=真实姓名，同义词=真实姓名、real_name
- `email`：邮箱地址，角色=OTHER，说明=邮箱地址，同义词=邮箱地址、email
- `mobile`：手机号，角色=OTHER，说明=手机号，同义词=手机号、mobile
- `operator_type_code`：员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工，角色=DIMENSION，说明=员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工，同义词=员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工、operator_type_code
- `operator_type_name`：员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工，角色=DIMENSION，说明=员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工，同义词=员工类型 1全渠道员工 2呼叫中心员工 3即时通讯员工 4工单员工 5外呼员工、operator_type_name
- `spu_name`：spu名称，角色=DIMENSION，说明=spu名称，同义词=spu名称、spu_name
- `level1_category_code`：一级类目code，角色=DIMENSION，说明=一级类目code，同义词=一级类目code、level1_category_code
- `level1_category_name`：一级类目name，角色=DIMENSION，说明=一级类目name，同义词=一级类目name、level1_category_name
- `level2_category_code`：二级类目code，角色=DIMENSION，说明=二级类目code，同义词=二级类目code、level2_category_code
- `level2_category_name`：二级类目name，角色=DIMENSION，说明=二级类目name，同义词=二级类目name、level2_category_name
- `level3_category_code`：三级类目code，角色=DIMENSION，说明=三级类目code，同义词=三级类目code、level3_category_code
- `level3_category_name`：三级类目name，角色=DIMENSION，说明=三级类目name，同义词=三级类目name、level3_category_name
- `brand_code`：品牌code，角色=DIMENSION，说明=品牌code，同义词=品牌code、brand_code
- `brand_name`：品牌name，角色=DIMENSION，说明=品牌name，同义词=品牌name、brand_name
