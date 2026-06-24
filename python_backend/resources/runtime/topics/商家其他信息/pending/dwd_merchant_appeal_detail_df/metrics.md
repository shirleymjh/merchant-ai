# 指标公式

- 申诉次数：COUNT(DISTINCT appeal_id)，单位=次，来源字段=appeal_id，同义词=申诉次数
- 申诉通过次数：SUM(CASE WHEN appeal_status_code = 1 OR appeal_status_name = '通过' THEN 1 ELSE 0 END)，单位=次，来源字段=appeal_status_code、appeal_status_name，同义词=申诉通过次数
- 申诉驳回次数：SUM(CASE WHEN appeal_status_code = 2 OR appeal_status_name = '驳回' THEN 1 ELSE 0 END)，单位=次，来源字段=appeal_status_code、appeal_status_name，同义词=申诉驳回次数
- 申诉成功率：appeal_pass_cnt / NULLIF(appeal_cnt, 0)，单位=%，来源指标=appeal_pass_cnt、appeal_cnt，来源字段=appeal_status_code、appeal_status_name、appeal_id，同义词=申诉成功率、申诉通过率、appeal_success_rate、appeal_pass_rate
- 处罚类申诉次数：SUM(CASE WHEN apply_type_code = 6 OR apply_type_name = '处罚' THEN 1 ELSE 0 END)，单位=次，来源字段=apply_type_code、apply_type_name，同义词=处罚类申诉次数
