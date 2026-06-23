CREATE TABLE IF NOT EXISTS merchant_ai_answer (
  `id` VARCHAR(64) NOT NULL COMMENT '每一次对话信息id',
  `question` STRING NOT NULL COMMENT '用户提问信息',
  `answer` STRING NOT NULL COMMENT '模型回复信息',
  `is_adopted` TINYINT NOT NULL DEFAULT '0' COMMENT '用户是否点击采纳',
  `like_flag` TINYINT NOT NULL DEFAULT '0' COMMENT '点赞',
  `dislike_flag` TINYINT NOT NULL DEFAULT '0' COMMENT '点踩',
  `merchant_id` VARCHAR(128) NOT NULL COMMENT '商家id',
  `merchant_name` VARCHAR(255) NOT NULL COMMENT '商家名称',
  `question_category_name` VARCHAR(128) NOT NULL COMMENT '问题分类名称',
  `doris_tables` VARCHAR(1024) NOT NULL COMMENT '调用的doris数据表',
  `suggested_questions` VARCHAR(2048) NOT NULL DEFAULT '[]' COMMENT '猜你想问',
  `langfuse_trace_id` VARCHAR(128) NOT NULL DEFAULT '' COMMENT 'Langfuse trace id',
  `langfuse_session_id` VARCHAR(128) NOT NULL DEFAULT '' COMMENT 'Langfuse session id',
  `create_time` DATETIME NOT NULL COMMENT '创建时间',
  `modify_time` DATETIME NOT NULL COMMENT '变更时间'
) ENGINE=OLAP
UNIQUE KEY(`id`)
COMMENT 'yshopping商家AI问答记录表'
DISTRIBUTED BY HASH(`id`) BUCKETS 4
PROPERTIES (
  'replication_allocation' = 'tag.location.default: 1',
  'enable_unique_key_merge_on_write' = 'true',
  'light_schema_change' = 'true'
);
