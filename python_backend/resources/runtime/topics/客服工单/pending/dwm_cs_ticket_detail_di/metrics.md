# 指标公式

- 催单次数：SUM(reminder_cnt)，单位=单，来源字段=reminder_cnt，同义词=催单次数、reminder_cnt
- 工单评价分数：AVG(ticket_score)，单位=-，来源字段=ticket_score，同义词=工单评价分数、ticket_score
- 工单量：COUNT(DISTINCT ticket_id)，单位=单，来源字段=ticket_id，同义词=工单量
