# Metrics

- 客服工单明细催单次数：SUM(reminder_cnt)，原始自然名=催单次数，粒度=ticket_detail，单位=单，来源字段=reminder_cnt，同义词=客服工单明细催单次数、reminder_cnt、客服工单催单次数
- 客服工单明细评价分数：AVG(ticket_score)，原始自然名=工单评价分数，粒度=ticket_detail，来源字段=ticket_score，同义词=客服工单明细评价分数、ticket_score、客服工单评价分数
- 客服工单明细量：COUNT(DISTINCT ticket_id)，原始自然名=工单量，粒度=ticket_detail，单位=单，来源字段=ticket_id，同义词=客服工单明细量、ticket_cnt、客服工单量高、客服工单明细数、按商品客服工单量，说明=按工单号去重统计工单数量
- 商品维度工单率：工单量 / 订单量，原始自然名=商品工单率，粒度=ticket_detail，单位=%，来源字段=ticket_cnt、order_detail_cnt，同义词=商品维度工单率、ticket_rate、客服工单率，说明=工单量与订单量的比值
