# 指标公式

- 入库数量：SUM(inbound_cnt)，单位=单，来源字段=inbound_cnt，同义词=入库数量、inbound_cnt
- 质检图片数量：SUM(photo_cnt)，单位=单，来源字段=photo_cnt，同义词=质检图片数量、photo_cnt
- 审核次数：SUM(check_cnt)，单位=单，来源字段=check_cnt，同义词=审核次数、check_cnt
- 首次质检结果：AVG(first_check_is_operate_pass)，单位=%，来源字段=first_check_is_operate_pass，同义词=首次质检结果、first_check_is_operate_pass、首次质检结果,不通过=0,通过=1
- 发货超时单量：COUNT(DISTINCT CASE WHEN outbound_modify_time > outbound_latest_time THEN outbound_id END)，单位=单，来源字段=outbound_id、outbound_modify_time、outbound_latest_time，同义词=发货超时单量
- 质检单量：COUNT(DISTINCT check_id)，单位=单，来源字段=check_id，同义词=质检单量
- 出库量：COUNT(DISTINCT outbound_id)，单位=单，来源字段=outbound_id，同义词=出库量
- 鉴定量：SUM(CASE WHEN identify_result_name IS NULL OR identify_result_name = '' THEN 0 ELSE inbound_cnt END)，单位=件，来源字段=identify_result_name、inbound_cnt，同义词=鉴定量
- 鉴定为假货量：SUM(CASE WHEN identify_result_code = 2 OR identify_result_name = '鉴定为假' THEN 1 ELSE 0 END)，单位=件，来源字段=identify_result_code、identify_result_name，同义词=鉴定为假货量
