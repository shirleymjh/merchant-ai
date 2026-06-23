# 指标公式

- 发售价分：AVG(spu_auth_price)，单位=分，来源字段=spu_auth_price，同义词=发售价分、spu_auth_price
- 商品数：COUNT(DISTINCT spu_id)，单位=个，来源字段=spu_id，同义词=商品数
- 商品审核通过量：SUM(CASE WHEN is_audit_pass = 1 THEN 1 ELSE 0 END)，单位=个，来源字段=is_audit_pass，同义词=商品审核通过量
- 商品审核拒绝量：SUM(CASE WHEN is_audit_pass = 0 THEN 1 ELSE 0 END)，单位=个，来源字段=is_audit_pass，同义词=商品审核拒绝量
- 上架商品量：SUM(CASE WHEN spu_status_code = 1 OR spu_status_name = '上架' THEN 1 ELSE 0 END)，单位=个，来源字段=spu_status_code、spu_status_name，同义词=上架商品量
