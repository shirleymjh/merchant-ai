CREATE TABLE `dwd_merchant_appeal_detail_df` (
  `appeal_id` bigint NOT NULL COMMENT "自增ID",
  `create_time` varchar(19) NOT NULL COMMENT "创建时间",
  `modify_time` varchar(19) NOT NULL COMMENT "修改时间",
  `spu_id` varchar(128) NOT NULL COMMENT "spu_id",
  `spu_name` varchar(255) NOT NULL COMMENT "spu名称",
  `level1_category_code` bigint NOT NULL COMMENT "一级类目code",
  `level1_category_name` varchar(128) NOT NULL COMMENT "一级类目name",
  `level2_category_code` bigint NOT NULL COMMENT "二级类目code",
  `level2_category_name` varchar(128) NOT NULL COMMENT "二级类目name",
  `level3_category_code` bigint NOT NULL COMMENT "三级类目code",
  `reason` varchar(2048) NOT NULL COMMENT "申诉文本",
  `images_url` varchar(2048) NOT NULL COMMENT "申诉图片",
  `appeal_status_code` bigint NOT NULL COMMENT "申诉状态code 1通过2驳回3取消",
  `appeal_status_name` varchar(128) NOT NULL COMMENT "申诉状态name 1通过2驳回3取消",
  `merchant_id` bigint NOT NULL COMMENT "商家id",
  `apply_type_code` bigint NOT NULL COMMENT "申诉类型code 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚",
  `apply_type_name` varchar(128) NOT NULL COMMENT "申诉类型name 1商品管理 2商家信息 3提现 4保证金 5供应链 6处罚",
  `pt` date NOT NULL COMMENT "日期分区yyyyMMdd",
  INDEX idx_merchant_id (`merchant_id`) USING INVERTED COMMENT "商家id倒排索引"
) ENGINE=OLAP
DUPLICATE KEY(`appeal_id`)
COMMENT 'dwd-商家域-商家申诉表'
PARTITION BY RANGE(`pt`)
(PARTITION p20260521 VALUES [('2026-05-21'), ('2026-05-22')),
PARTITION p20260522 VALUES [('2026-05-22'), ('2026-05-23')),
PARTITION p20260523 VALUES [('2026-05-23'), ('2026-05-24')),
PARTITION p20260524 VALUES [('2026-05-24'), ('2026-05-25')),
PARTITION p20260525 VALUES [('2026-05-25'), ('2026-05-26')))
DISTRIBUTED BY HASH(`appeal_id`) BUCKETS 8
PROPERTIES (
"replication_allocation" = "tag.location.default: 1",
"min_load_replica_num" = "-1",
"is_being_synced" = "false",
"dynamic_partition.enable" = "true",
"dynamic_partition.time_unit" = "DAY",
"dynamic_partition.time_zone" = "Asia/Shanghai",
"dynamic_partition.start" = "-3",
"dynamic_partition.end" = "1",
"dynamic_partition.prefix" = "p",
"dynamic_partition.replication_allocation" = "tag.location.default: 1",
"dynamic_partition.buckets" = "8",
"dynamic_partition.create_history_partition" = "true",
"dynamic_partition.history_partition_num" = "3",
"dynamic_partition.hot_partition_num" = "0",
"dynamic_partition.reserved_history_periods" = "NULL",
"dynamic_partition.storage_policy" = "",
"storage_medium" = "hdd",
"storage_format" = "V2",
"inverted_index_storage_format" = "V3",
"light_schema_change" = "true",
"disable_auto_compaction" = "false",
"enable_single_replica_compaction" = "false",
"group_commit_interval_ms" = "10000",
"group_commit_data_bytes" = "134217728"
);
