# Metrics

- 申诉明细次数：COUNT(DISTINCT appeal_id)，原始自然名=申诉次数，粒度=appeal_detail，单位=次，来源字段=appeal_id，同义词=申诉明细次数、appeal_cnt，说明=按申诉 ID 去重统计商家申诉次数
- 申诉明细通过次数：SUM(CASE WHEN appeal_status_code = 1 OR appeal_status_name = '通过' THEN 1 ELSE 0 END)，原始自然名=申诉通过次数，粒度=appeal_detail，单位=次，来源字段=appeal_status_code、appeal_status_name，同义词=申诉明细通过次数、appeal_pass_cnt，说明=按申诉状态统计通过次数
- 申诉明细驳回次数：SUM(CASE WHEN appeal_status_code = 2 OR appeal_status_name = '驳回' THEN 1 ELSE 0 END)，原始自然名=申诉驳回次数，粒度=appeal_detail，单位=次，来源字段=appeal_status_code、appeal_status_name，同义词=申诉明细驳回次数、appeal_reject_cnt，说明=按申诉状态统计驳回次数
- 申诉明细处罚类申诉次数：SUM(CASE WHEN apply_type_code = 6 OR apply_type_name = '处罚' THEN 1 ELSE 0 END)，原始自然名=处罚类申诉次数，粒度=appeal_detail，单位=次，来源字段=apply_type_code、apply_type_name，同义词=申诉明细处罚类申诉次数、punish_appeal_cnt、申诉处罚类申诉次数，说明=按申诉类型统计处罚类申诉次数
