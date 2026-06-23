CREATE TABLE `dwd_merchant_deposit_recharge_df` (
  `create_time` varchar(19) NOT NULL COMMENT "创建时间",
  `modify_time` varchar(19) NOT NULL COMMENT "修改时间",
  `merchant_id` varchar(128) NOT NULL COMMENT "商家id",
  `user_id` varchar(128) NOT NULL COMMENT "用户id",
  `deposit_recharge_id` varchar(128) NOT NULL COMMENT "补缴单单号(充值申请号)",
  `trans_id` varchar(128) NOT NULL COMMENT "交易流水号",
  `currency` varchar(32) NOT NULL COMMENT "币种",
  `deposit_recharge_amt` bigint NOT NULL COMMENT "金额, 通用单位元",
  `trans_voucher` varchar(1024) NOT NULL COMMENT "交易流水凭证",
  `remark` varchar(1024) NOT NULL COMMENT "备注",
  `pt` date NOT NULL COMMENT "日期分区yyyyMMdd",
  INDEX idx_merchant_id (`merchant_id`) USING INVERTED COMMENT "商家id倒排索引"
) ENGINE=OLAP
DUPLICATE KEY(`create_time`)
COMMENT 'dwd-商家域-商家保证金充值表'
PARTITION BY RANGE(`pt`)
(PARTITION p20260521 VALUES [('2026-05-21'), ('2026-05-22')),
PARTITION p20260522 VALUES [('2026-05-22'), ('2026-05-23')),
PARTITION p20260523 VALUES [('2026-05-23'), ('2026-05-24')),
PARTITION p20260524 VALUES [('2026-05-24'), ('2026-05-25')),
PARTITION p20260525 VALUES [('2026-05-25'), ('2026-05-26')))
DISTRIBUTED BY HASH(`deposit_recharge_id`) BUCKETS 8
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
