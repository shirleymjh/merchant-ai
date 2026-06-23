# 商品管理 / dwm_goods_detail_df

状态：PENDING_REVIEW
表说明：dwm-商品系统-商品上新全流程-全量表
数据粒度：商品/SPU 全流程快照粒度
时间字段：`pt`
商家过滤字段：`seller_id`
人工业务说明：商品上新全流程全量表，承载商品审核、拒绝、通过、上架、类目和价格信息

## 列级语义
- `spu_id`：spu ID，角色=KEY，说明=spu ID，同义词=spu ID、spu_id、商品
- `spu_name`：商品名称，角色=DIMENSION，说明=商品名称，同义词=商品名称、spu_name、商品
- `seller_id`：spu发布人id，角色=KEY，说明=spu发布人id，同义词=spu发布人id、seller_id
- `seller_name`：spu发布人name，角色=DIMENSION，说明=spu发布人name，同义词=spu发布人name、seller_name
- `source_type_code`：spu来源code，角色=DIMENSION，说明=spu来源code，同义词=spu来源code、source_type_code
- `source_type_name`：spu来源name 个人卖家 企业卖家，角色=DIMENSION，说明=spu来源name 个人卖家 企业卖家，同义词=spu来源name 个人卖家 企业卖家、source_type_name
- `level1_category_code`：一级类目code，角色=DIMENSION，说明=一级类目code，同义词=一级类目code、level1_category_code
- `level1_category_name`：一级类目name，角色=DIMENSION，说明=一级类目name，同义词=一级类目name、level1_category_name
- `level2_category_code`：二级类目code，角色=DIMENSION，说明=二级类目code，同义词=二级类目code、level2_category_code
- `level2_category_name`：二级类目name，角色=DIMENSION，说明=二级类目name，同义词=二级类目name、level2_category_name
- `level3_category_code`：三级类目code，角色=DIMENSION，说明=三级类目code，同义词=三级类目code、level3_category_code
- `level3_category_name`：三级类目name，角色=DIMENSION，说明=三级类目name，同义词=三级类目name、level3_category_name
- `brand_code`：品牌code，角色=DIMENSION，说明=品牌code，同义词=品牌code、brand_code
- `brand_name`：品牌name，角色=DIMENSION，说明=品牌name，同义词=品牌name、brand_name
- `fit_code`：适用人群code，角色=DIMENSION，说明=适用人群code，同义词=适用人群code、fit_code
- `fit_name`：适用人群name，角色=DIMENSION，说明=适用人群name，同义词=适用人群name、fit_name
- `spu_auth_price`：发售价分，角色=METRIC，说明=发售价分，公式=AVG(spu_auth_price)，同义词=发售价分、spu_auth_price、商品
- `spu_logo_url`：logo图，角色=OTHER，说明=logo图，同义词=logo图、spu_logo_url、商品
- `spu_video_url`：视频图，角色=OTHER，说明=视频图，同义词=视频图、spu_video_url、商品
- `article_id`：货号，角色=KEY，说明=货号，同义词=货号、article_id
- `spu_status_code`：商品状态code 0.下架 1.上架 2.待提交 3.待审核 4.审核通过，角色=DIMENSION，说明=商品状态code 0.下架 1.上架 2.待提交 3.待审核 4.审核通过，同义词=商品状态code 0.下架 1.上架 2.待提交 3.待审核 4.审核通过、spu_status_code、商品
- `spu_status_name`：商品状态name 0.下架 1.上架 2.待提交 3.待审核 4.审核通过，角色=DIMENSION，说明=商品状态name 0.下架 1.上架 2.待提交 3.待审核 4.审核通过，同义词=商品状态name 0.下架 1.上架 2.待提交 3.待审核 4.审核通过、spu_status_name、商品
- `spu_desc`：商品描述，角色=OTHER，说明=商品描述，同义词=商品描述、spu_desc、商品
- `spu_apply_create_time`：创建时间，角色=TIME，说明=创建时间，同义词=创建时间、spu_apply_create_time、商品
- `spu_apply_modify_time`：变更时间，角色=TIME，说明=变更时间，同义词=变更时间、spu_apply_modify_time、商品
- `biz_id`：唯一ID，角色=KEY，说明=唯一ID，同义词=唯一ID、biz_id
- `completion_operate_type_code`：任务状态code。反馈问题，角色=DIMENSION，说明=任务状态code。反馈问题；提交任务；，同义词=任务状态code。反馈问题、completion_operate_type_code、任务状态code。反馈问题；提交任务；
- `completion_operate_type_name`：任务状态name。反馈问题，角色=DIMENSION，说明=任务状态name。反馈问题；提交任务；，同义词=任务状态name。反馈问题、completion_operate_type_name、任务状态name。反馈问题；提交任务；
- `is_completion_pass`：是否通过，角色=DIMENSION，说明=是否通过，同义词=是否通过、is_completion_pass
- `completion_remark`：备注，角色=OTHER，说明=备注，同义词=备注、completion_remark
- `first_completion_operate_id`：首次操作人，角色=KEY，说明=首次操作人，同义词=首次操作人、first_completion_operate_id
- `first_completion_operate_name`：首次操作人name，角色=DIMENSION，说明=首次操作人name，同义词=首次操作人name、first_completion_operate_name
- `completion_operate_id`：最新操作人，角色=KEY，说明=最新操作人，同义词=最新操作人、completion_operate_id
- `completion_operate_name`：最新操作人name，角色=DIMENSION，说明=最新操作人name，同义词=最新操作人name、completion_operate_name
- `first_completion_operate_time`：首次操作时间，角色=TIME，说明=首次操作时间，同义词=首次操作时间、first_completion_operate_time
- `completion_operate_time`：操作时间，角色=TIME，说明=操作时间，同义词=操作时间、completion_operate_time
- `audit_operate_type_code`：任务状态code。反馈问题，角色=DIMENSION，说明=任务状态code。反馈问题；提交任务；，同义词=任务状态code。反馈问题、audit_operate_type_code、任务状态code。反馈问题；提交任务；
- `audit_operate_type_name`：任务状态name。反馈问题，角色=DIMENSION，说明=任务状态name。反馈问题；提交任务；，同义词=任务状态name。反馈问题、audit_operate_type_name、任务状态name。反馈问题；提交任务；
- `is_audit_pass`：是否通过，角色=DIMENSION，说明=是否通过，同义词=是否通过、is_audit_pass
- `audit_remark`：备注，角色=OTHER，说明=备注，同义词=备注、audit_remark
- `first_audit_operate_id`：首次操作人，角色=KEY，说明=首次操作人，同义词=首次操作人、first_audit_operate_id
- `first_audit_operate_name`：首次操作人name，角色=DIMENSION，说明=首次操作人name，同义词=首次操作人name、first_audit_operate_name
- `first_audit_operate_supplier_name`：首次操作人供应商名称，角色=DIMENSION，说明=首次操作人供应商名称，同义词=首次操作人供应商名称、first_audit_operate_supplier_name
- `audit_operate_id`：最新操作人，角色=KEY，说明=最新操作人，同义词=最新操作人、audit_operate_id
- `audit_operate_name`：最新操作人name，角色=DIMENSION，说明=最新操作人name，同义词=最新操作人name、audit_operate_name
- `audit_operate_supplier_name`：最新操作人供应商名称，角色=DIMENSION，说明=最新操作人供应商名称，同义词=最新操作人供应商名称、audit_operate_supplier_name
- `first_audit_operate_time`：首次操作时间，角色=TIME，说明=首次操作时间，同义词=首次操作时间、first_audit_operate_time
- `audit_operate_time`：操作时间，角色=TIME，说明=操作时间，同义词=操作时间、audit_operate_time
- `import_task_id`：导入任务id，角色=KEY，说明=导入任务id，同义词=导入任务id、import_task_id
- `excel_url`：excel存放路径，角色=OTHER，说明=excel存放路径，同义词=excel存放路径、excel_url
- `excel_name`：导入excel名称，角色=DIMENSION，说明=导入excel名称，同义词=导入excel名称、excel_name
- `is_import_pass`：导入状态，角色=DIMENSION，说明=导入状态，同义词=导入状态、is_import_pass
- `import_time`：导入时间，角色=TIME，说明=导入时间，同义词=导入时间、import_time
- `import_operate_id`：操作人id，角色=KEY，说明=操作人id，同义词=操作人id、import_operate_id
- `import_operate_name`：操作人name，角色=DIMENSION，说明=操作人name，同义词=操作人name、import_operate_name
- `is_risk_audit_pass`：是否风控审核通过，角色=DIMENSION，说明=是否风控审核通过，同义词=是否风控审核通过、is_risk_audit_pass
- `machine_audit_time`：机审时间，角色=TIME，说明=机审时间，同义词=机审时间、machine_audit_time
- `is_manual`：机审是否转人工，角色=DIMENSION，说明=机审是否转人工，同义词=机审是否转人工、is_manual
- `first_risk_tag_user_id`：首次risk最新审核人ID，角色=KEY，说明=首次risk最新审核人ID，同义词=首次risk最新审核人ID、first_risk_tag_user_id
- `first_risk_tag_user_name`：首次risk最新审核人name，角色=DIMENSION，说明=首次risk最新审核人name，同义词=首次risk最新审核人name、first_risk_tag_user_name
- `risk_tag_user_id`：risk最新审核人ID，角色=KEY，说明=risk最新审核人ID，同义词=risk最新审核人ID、risk_tag_user_id
- `risk_tag_user_name`：risk最新审核人名称，角色=DIMENSION，说明=risk最新审核人名称，同义词=risk最新审核人名称、risk_tag_user_name
- `risk_operate_supplier_name`：risk最新审核人供应商，角色=DIMENSION，说明=risk最新审核人供应商，同义词=risk最新审核人供应商、risk_operate_supplier_name
- `is_white`：是否加白，角色=DIMENSION，说明=是否加白，同义词=是否加白、is_white
- `pt`：业务日期，角色=TIME，说明=业务日期，同义词=业务日期、pt
