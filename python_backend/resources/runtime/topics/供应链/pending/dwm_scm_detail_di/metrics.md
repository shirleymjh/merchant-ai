# Metrics

- 供应链明细入库数量：SUM(inbound_cnt)，原始自然名=入库数量，粒度=supply_chain_detail，单位=单，来源字段=inbound_cnt，同义词=供应链明细入库数量、inbound_cnt、供应链入库数量、供应链入库量
- 供应链明细质检图片数量：SUM(photo_cnt)，原始自然名=质检图片数量，粒度=supply_chain_detail，单位=单，来源字段=photo_cnt，同义词=供应链明细质检图片数量、photo_cnt、供应链质检图片数量
- 供应链明细审核次数：SUM(check_cnt)，原始自然名=审核次数，粒度=supply_chain_detail，单位=单，来源字段=check_cnt，同义词=供应链明细审核次数、check_cnt、供应链审核次数
- 供应链明细首次质检结果：AVG(first_check_is_operate_pass)，原始自然名=首次质检结果，粒度=supply_chain_detail，单位=%，来源字段=first_check_is_operate_pass，同义词=供应链明细首次质检结果、first_check_is_operate_pass、供应链首次质检结果、首次质检结果,不通过=0,通过=1，说明=首次质检结果,不通过=0,通过=1
- 供应链明细发货超时单量：COUNT(DISTINCT CASE WHEN outbound_modify_time > outbound_latest_time THEN outbound_id END)，原始自然名=发货超时单量，粒度=supply_chain_detail，单位=单，来源字段=outbound_id、outbound_modify_time、outbound_latest_time，同义词=供应链明细发货超时单量、ship_timeout_detail_cnt、供应链发货超时单量，说明=出库变更时间晚于最晚出库时间的发货超时单量
- 供应链明细质检单量：COUNT(DISTINCT check_id)，原始自然名=质检单量，粒度=supply_chain_detail，单位=单，来源字段=check_id，同义词=供应链明细质检单量、quality_check_cnt、供应链质检单量，说明=按质检 ID 去重统计质检单量
- 供应链明细出库量：COUNT(DISTINCT outbound_id)，原始自然名=出库量，粒度=supply_chain_detail，单位=单，来源字段=outbound_id，同义词=供应链明细出库量、scm_outbound_cnt、供应链出库量，说明=按出库 ID 去重统计出库单量
- 供应链明细鉴定量：SUM(CASE WHEN identify_result_name IS NULL OR identify_result_name = '' THEN 0 ELSE inbound_cnt END)，原始自然名=鉴定量，粒度=supply_chain_detail，单位=件，来源字段=identify_result_name、inbound_cnt，同义词=供应链明细鉴定量、scm_identify_cnt、供应链鉴定量，说明=按鉴定结果统计发生鉴定的入库数量
- 供应链明细鉴定为假货量：SUM(CASE WHEN identify_result_code = 2 OR identify_result_name = '鉴定为假' THEN 1 ELSE 0 END)，原始自然名=鉴定为假货量，粒度=supply_chain_detail，单位=件，来源字段=identify_result_code、identify_result_name，同义词=供应链明细鉴定为假货量、fake_identify_detail_cnt、供应链鉴定为假货量，说明=按鉴定结果统计假货量
