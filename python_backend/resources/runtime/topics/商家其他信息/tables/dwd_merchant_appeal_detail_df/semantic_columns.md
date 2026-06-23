# 商家其他信息 / dwd_merchant_appeal_detail_df

状态：PUBLISHED
表说明：dwd-商家域-商家申诉表
数据粒度：商家申诉明细粒度
时间字段：`pt`
商家过滤字段：`merchant_id`
人工业务说明：商家申诉明细表，承载申诉状态、申诉类型以及处罚类申诉判断

## 列级语义
- `appeal_id`：自增ID，角色=KEY，说明=自增ID，同义词=自增ID、appeal_id、申诉、商家申诉
- `create_time`：创建时间，角色=TIME，说明=创建时间，同义词=创建时间、create_time
- `modify_time`：修改时间，角色=TIME，说明=修改时间，同义词=修改时间、modify_time
- `spu_id`：spu_id，角色=KEY，说明=spu_id，同义词=spu_id、商品
- `spu_name`：spu名称，角色=DIMENSION，说明=spu名称，同义词=spu名称、spu_name、商品
- `level1_category_code`：一级类目code，角色=DIMENSION，说明=一级类目code，同义词=一级类目code、level1_category_code
- `level1_category_name`：一级类目name，角色=DIMENSION，说明=一级类目name，同义词=一级类目name、level1_category_name
- `level2_category_code`：二级类目code，角色=DIMENSION，说明=二级类目code，同义词=二级类目code、level2_category_code
- `level2_category_name`：二级类目name，角色=DIMENSION，说明=二级类目name，同义词=二级类目name、level2_category_name
- `level3_category_code`：三级类目code，角色=DIMENSION，说明=三级类目code，同义词=三级类目code、level3_category_code
- `reason`：申诉文本，角色=DIMENSION，说明=申诉文本，同义词=申诉文本、reason、申诉、商家申诉
- `images_url`：申诉图片，角色=OTHER，说明=申诉图片，同义词=申诉图片、images_url、申诉、商家申诉
- `appeal_status_code`：申诉状态code 1通过2驳回3取消，角色=DIMENSION，说明=申诉状态code 1通过2驳回3取消，同义词=申诉状态code 1通过2驳回3取消、appeal_status_code、申诉、商家申诉
- `appeal_status_name`：申诉状态name 1通过2驳回3取消，角色=DIMENSION，说明=申诉状态name 1通过2驳回3取消，同义词=申诉状态name 1通过2驳回3取消、appeal_status_name、申诉、商家申诉
- `merchant_id`：商家id，角色=KEY，说明=商家id，同义词=商家id、merchant_id
- `apply_type_code`：申诉类型code 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚，角色=DIMENSION，说明=申诉类型code 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚，同义词=申诉类型code 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚、apply_type_code、保证金、押金、申诉、商家申诉、商品
- `apply_type_name`：申诉类型name 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚，角色=DIMENSION，说明=申诉类型name 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚，同义词=申诉类型name 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚、apply_type_name、保证金、押金、申诉、商家申诉、商品
- `pt`：日期分区yyyyMMdd，角色=TIME，说明=日期分区yyyyMMdd，同义词=日期分区yyyyMMdd、pt
