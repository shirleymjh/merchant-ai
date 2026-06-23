CREATE TABLE IF NOT EXISTS merchant_memory_fact (
  fact_id VARCHAR(64) NOT NULL COMMENT '记忆事实id',
  merchant_id VARCHAR(128) NOT NULL COMMENT '商家id',
  category VARCHAR(64) NOT NULL DEFAULT '' COMMENT '适用业务域',
  fact_type VARCHAR(64) NOT NULL DEFAULT '' COMMENT '事实类型，如 QUERY_PREFERENCE/CORRECTION/REINFORCEMENT',
  content STRING NOT NULL COMMENT '结构化长期记忆内容',
  confidence DOUBLE NOT NULL DEFAULT '0.7' COMMENT '置信度',
  source_qa_id VARCHAR(64) NOT NULL DEFAULT '' COMMENT '来源问答id',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  updated_at DATETIME NOT NULL COMMENT '更新时间'
) ENGINE=OLAP
UNIQUE KEY(`fact_id`)
COMMENT '商家长期记忆事实表'
DISTRIBUTED BY HASH(`fact_id`) BUCKETS 4
PROPERTIES (
  'replication_allocation' = 'tag.location.default: 1',
  'enable_unique_key_merge_on_write' = 'true',
  'light_schema_change' = 'true'
);
