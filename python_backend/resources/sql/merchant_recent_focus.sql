CREATE TABLE IF NOT EXISTS merchant_recent_focus (
  merchant_id VARCHAR(128) NOT NULL COMMENT '商家id',
  top_categories_json STRING NOT NULL COMMENT '近期关注类目',
  top_metrics_json STRING NOT NULL COMMENT '近期高频指标',
  common_time_ranges_json STRING NOT NULL COMMENT '常用时间范围',
  focus_pattern VARCHAR(512) NOT NULL DEFAULT '' COMMENT '近期关注摘要',
  last_active_at DATETIME NULL COMMENT '最近一次活跃时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间'
) ENGINE=OLAP
UNIQUE KEY(`merchant_id`)
COMMENT '商家近期关注摘要表'
DISTRIBUTED BY HASH(`merchant_id`) BUCKETS 4
PROPERTIES (
  'replication_allocation' = 'tag.location.default: 1',
  'enable_unique_key_merge_on_write' = 'true',
  'light_schema_change' = 'true'
);
