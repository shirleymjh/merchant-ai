CREATE TABLE IF NOT EXISTS merchant_memory_event (
  merchant_id VARCHAR(128) NOT NULL COMMENT '商家id',
  created_at DATETIME NOT NULL COMMENT '创建时间',
  qa_id VARCHAR(64) NOT NULL COMMENT '本次问答id',
  question STRING NOT NULL COMMENT '原始问题',
  category VARCHAR(64) NOT NULL COMMENT '业务类目',
  answer_mode VARCHAR(32) NOT NULL COMMENT '回答模式',
  metric_column VARCHAR(128) NOT NULL DEFAULT '' COMMENT '指标字段',
  metric_name VARCHAR(128) NOT NULL DEFAULT '' COMMENT '指标名称',
  time_range VARCHAR(64) NOT NULL DEFAULT '' COMMENT '时间范围',
  is_followup TINYINT NOT NULL DEFAULT '0' COMMENT '是否连续追问',
  is_analysis TINYINT NOT NULL DEFAULT '0' COMMENT '是否分析类问题',
  is_adopted TINYINT NOT NULL DEFAULT '0' COMMENT '是否已采纳',
  event_weight INT NOT NULL DEFAULT '1' COMMENT '事件基础权重'
) ENGINE=OLAP
DUPLICATE KEY(`merchant_id`, `created_at`, `qa_id`)
COMMENT '商家问答记忆事件表'
DISTRIBUTED BY HASH(`merchant_id`) BUCKETS 4
PROPERTIES (
  'replication_allocation' = 'tag.location.default: 1',
  'light_schema_change' = 'true'
);
