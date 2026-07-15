# Metrics

- 商品明细发售价分：AVG(spu_auth_price)，原始自然名=发售价分，粒度=goods_detail，单位=分，来源字段=spu_auth_price，同义词=商品明细发售价分、spu_auth_price、商品发售价分、按商品发售价分、按SPU发售价分
- 商品明细去重商品数：COUNT(DISTINCT spu_id)，原始自然名=商品数，粒度=goods_detail，单位=个，来源字段=spu_id，同义词=商品明细去重商品数、goods_cnt、商品去重商品数、按商品数、按SPU数，说明=按 SPU 去重统计商品数量
- 商品审核通过明细量：SUM(CASE WHEN is_audit_pass = 1 THEN 1 ELSE 0 END)，原始自然名=商品审核通过量，粒度=goods_detail，单位=个，来源字段=is_audit_pass，同义词=商品审核通过明细量、goods_audit_pass_detail_cnt、按商品审核通过量、按SPU审核通过量、审核通过商品量、审核通过商品，说明=按商品审核是否通过统计通过量
- 商品审核拒绝明细量：SUM(CASE WHEN is_audit_pass = 0 THEN 1 ELSE 0 END)，原始自然名=商品审核拒绝量，粒度=goods_detail，单位=个，来源字段=is_audit_pass，同义词=商品审核拒绝明细量、goods_audit_reject_detail_cnt、按商品审核拒绝量、按SPU审核拒绝量、被拒商品量、被拒SPU量、驳回商品量、审核拒绝商品量，说明=按商品审核是否通过统计拒绝量
- 商品上架明细量：SUM(CASE WHEN spu_status_code = 1 OR spu_status_name = '上架' THEN 1 ELSE 0 END)，原始自然名=上架商品量，粒度=goods_detail，单位=个，来源字段=spu_status_code、spu_status_name，同义词=商品上架明细量、goods_online_detail_cnt、商品上架量、按商品上架商品量、按SPU上架商品量、发布成功商品量、发布成功商品、重新发布成功商品量、重新发布成功商品、重新发布成功、重新上架商品量、上架成功商品量，说明=按商品状态统计上架商品量
