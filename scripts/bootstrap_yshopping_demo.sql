CREATE DATABASE IF NOT EXISTS yshopping;
USE yshopping;

CREATE TABLE IF NOT EXISTS ads_merchant_profile (
  merchant_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  user_id VARCHAR(128),
  merchant_type_name VARCHAR(128),
  brand_type_name VARCHAR(128),
  balance_type_name VARCHAR(128),
  mobile VARCHAR(32),
  company_name VARCHAR(255),
  license_id VARCHAR(128),
  is_unconditional_refund TINYINT,
  is_invoice TINYINT,
  contact_name VARCHAR(128),
  business_address VARCHAR(512),
  send_address VARCHAR(512),
  refnd_address VARCHAR(512),
  bank_name VARCHAR(128),
  bank_account VARCHAR(128),
  account_type_name VARCHAR(128),
  poundage_discount DECIMAL(10, 4),
  deposit_amt DECIMAL(18, 2),
  order_cnt_1d BIGINT,
  order_user_cnt_1d BIGINT,
  order_gmv_amt_1d DECIMAL(18, 2),
  pay_order_cnt_1d BIGINT,
  pay_gmv_amt_1d DECIMAL(18, 2),
  trade_success_order_cnt_1d BIGINT,
  trade_success_gmv_amt_1d DECIMAL(18, 2),
  avg_pay_order_amt_1d DECIMAL(18, 2),
  ship_timeout_order_cnt_1d BIGINT,
  signed_order_cnt_1d BIGINT,
  delivery_timeout_order_cnt_1d BIGINT,
  refund_amt_1d DECIMAL(18, 2),
  return_success_amt_1d DECIMAL(18, 2),
  return_success_cnt_1d BIGINT,
  return_cnt_1d BIGINT,
  direct_refund_cnt_1d BIGINT,
  refund_rate_1d DECIMAL(10, 4),
  cs_ticket_cnt_1d BIGINT,
  ticket_reopen_cnt_1d BIGINT,
  ticket_reminder_cnt_1d BIGINT,
  ticket_close_cnt_1d BIGINT,
  avg_ticket_score_1d DECIMAL(10, 2),
  seller_repay_order_cnt_1d BIGINT,
  seller_repay_amt_1d DECIMAL(18, 2),
  pay_success_discount_order_cnt_1d BIGINT,
  pay_success_discount_amt_1d DECIMAL(18, 2),
  trade_success_discount_order_cnt_1d BIGINT,
  trade_success_discount_amt_1d DECIMAL(18, 2),
  pay_discount_rate_1d DECIMAL(10, 4),
  goods_audit_reject_cnt_1d BIGINT,
  goods_audit_pass_cnt_1d BIGINT,
  goods_online_cnt_1d BIGINT,
  goods_apply_cnt_1d BIGINT,
  deposit_pay_cnt_1d BIGINT,
  appeal_success_cnt_1d BIGINT,
  appeal_cnt_1d BIGINT,
  punish_cnt_1d BIGINT,
  scm_performance_cnt_1d BIGINT
) ENGINE=OLAP
UNIQUE KEY(merchant_id, pt)
DISTRIBUTED BY HASH(merchant_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dim_merchant_df (
  merchant_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  user_id VARCHAR(128),
  merchant_type_name VARCHAR(128),
  brand_type_name VARCHAR(128),
  balance_type_name VARCHAR(128),
  mobile VARCHAR(32),
  company_name VARCHAR(255),
  license_id VARCHAR(128),
  contact_name VARCHAR(128),
  business_address VARCHAR(512),
  send_address VARCHAR(512),
  refnd_address VARCHAR(512),
  bank_name VARCHAR(128),
  bank_account VARCHAR(128),
  account_type_name VARCHAR(128),
  ship_model_name VARCHAR(128),
  is_invoice TINYINT,
  is_unconditional_refund TINYINT,
  init_deposit_amt DECIMAL(18, 2),
  deposit_freeze DECIMAL(18, 2),
  deposit_amt DECIMAL(18, 2),
  min_poundage DECIMAL(10, 4),
  max_poundage DECIMAL(10, 4),
  poundage_discount DECIMAL(10, 4)
) ENGINE=OLAP
UNIQUE KEY(merchant_id, pt)
DISTRIBUTED BY HASH(merchant_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_trade_order_detail_di (
  sub_order_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  order_id VARCHAR(128),
  buyer_id VARCHAR(128),
  buyer_name VARCHAR(255),
  seller_id VARCHAR(128),
  sub_order_status_name VARCHAR(128),
  spu_id VARCHAR(128),
  spu_name VARCHAR(255),
  sku_id VARCHAR(128),
  sku_name VARCHAR(255),
  sku_cnt BIGINT,
  pay_amt DECIMAL(18, 2),
  pay_status_name VARCHAR(128),
  sub_order_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(sub_order_id, pt)
DISTRIBUTED BY HASH(sub_order_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_trade_refund_detail_di (
  refund_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  order_id VARCHAR(128),
  sub_order_id VARCHAR(128),
  seller_id VARCHAR(128),
  buyer_id VARCHAR(128),
  refund_status_name VARCHAR(128),
  refund_reason VARCHAR(255),
  sku_title VARCHAR(255),
  pay_amt DECIMAL(18, 2),
  refund_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(refund_id, pt)
DISTRIBUTED BY HASH(refund_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_cs_ticket_detail_di (
  ticket_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  seller_id VARCHAR(128),
  ticket_title VARCHAR(255),
  ticket_status_name VARCHAR(128),
  priority_name VARCHAR(128),
  is_reopen TINYINT,
  is_reminder TINYINT,
  ticket_score DECIMAL(10, 2),
  ticket_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(ticket_id, pt)
DISTRIBUTED BY HASH(ticket_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_cs_repay_detail_df (
  bill_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  seller_id VARCHAR(128),
  order_id VARCHAR(128),
  sub_order_id VARCHAR(128),
  repay_amt DECIMAL(18, 2),
  repay_status_name VARCHAR(128),
  pay_way_name VARCHAR(128),
  create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(bill_id, pt)
DISTRIBUTED BY HASH(bill_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_coupon_detail_di (
  coupon_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  seller_id VARCHAR(128),
  user_id VARCHAR(128),
  template_title VARCHAR(255),
  coupon_amt DECIMAL(18, 2),
  coupon_send_status_name VARCHAR(128),
  coupon_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(coupon_id, pt)
DISTRIBUTED BY HASH(coupon_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_goods_detail_df (
  spu_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  seller_id VARCHAR(128),
  spu_name VARCHAR(255),
  spu_status_name VARCHAR(128),
  audit_operate_type_name VARCHAR(128),
  is_audit_pass TINYINT,
  audit_remark VARCHAR(512),
  spu_apply_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(spu_id, pt)
DISTRIBUTED BY HASH(spu_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwm_scm_detail_di (
  inbound_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  seller_id VARCHAR(128),
  inbound_status_name VARCHAR(128),
  spu_id VARCHAR(128),
  sku_id VARCHAR(128),
  inbound_cnt BIGINT,
  warehouse_id VARCHAR(128),
  check_status_name VARCHAR(128),
  identify_result_name VARCHAR(128),
  outbound_id VARCHAR(128),
  inbound_create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(inbound_id, pt)
DISTRIBUTED BY HASH(inbound_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwd_merchant_deposit_recharge_df (
  deposit_recharge_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  merchant_id VARCHAR(128),
  trans_id VARCHAR(128),
  currency VARCHAR(32),
  deposit_recharge_amt DECIMAL(18, 2),
  remark VARCHAR(512),
  create_time DATETIME,
  modify_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(deposit_recharge_id, pt)
DISTRIBUTED BY HASH(deposit_recharge_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS dwd_merchant_appeal_detail_df (
  appeal_id VARCHAR(128) NOT NULL,
  pt DATE NOT NULL,
  merchant_id VARCHAR(128),
  spu_id VARCHAR(128),
  spu_name VARCHAR(255),
  appeal_status_name VARCHAR(128),
  apply_type_name VARCHAR(128),
  reason VARCHAR(512),
  create_time DATETIME
) ENGINE=OLAP
UNIQUE KEY(appeal_id, pt)
DISTRIBUTED BY HASH(appeal_id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

CREATE TABLE IF NOT EXISTS merchant_ai_answer (
  id VARCHAR(64) NOT NULL,
  question STRING NOT NULL,
  answer STRING NOT NULL,
  is_adopted TINYINT NOT NULL DEFAULT '0',
  like_flag TINYINT NOT NULL DEFAULT '0',
  dislike_flag TINYINT NOT NULL DEFAULT '0',
  merchant_id VARCHAR(128) NOT NULL,
  merchant_name VARCHAR(255) NOT NULL,
  question_category_name VARCHAR(128) NOT NULL,
  doris_tables VARCHAR(1024) NOT NULL,
  suggested_questions VARCHAR(2048) NOT NULL DEFAULT '[]',
  create_time DATETIME NOT NULL,
  modify_time DATETIME NOT NULL
) ENGINE=OLAP
UNIQUE KEY(id)
DISTRIBUTED BY HASH(id) BUCKETS 1
PROPERTIES (
  "replication_allocation" = "tag.location.default: 1",
  "enable_unique_key_merge_on_write" = "true"
);

INSERT INTO dim_merchant_df VALUES
('100', '2026-06-02', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', '商家自发货', 1, 1, 10000.00, 1200.00, 8800.00, 0.0030, 0.0080, 0.0060);

INSERT INTO ads_merchant_profile VALUES
('100', '2026-05-27', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 82, 64, 18650.00, 78, 18010.00, 73, 16980.00, 230.90, 3, 71, 2, 1420.00, 980.00, 5, 7, 2, 0.0897, 6, 1, 2, 5, 4.60, 1, 80.00, 9, 310.00, 7, 260.00, 0.0172, 2, 18, 126, 20, 1, 0, 1, 0, 14),
('100', '2026-05-28', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 96, 75, 22430.00, 92, 21670.00, 88, 20590.00, 235.54, 4, 85, 3, 2380.00, 1280.00, 7, 9, 3, 0.0978, 8, 0, 3, 7, 4.50, 2, 160.00, 11, 460.00, 8, 330.00, 0.0212, 3, 21, 139, 25, 0, 1, 2, 1, 18),
('100', '2026-05-29', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 105, 82, 25780.00, 101, 24920.00, 94, 23160.00, 246.73, 5, 90, 4, 2840.00, 1530.00, 9, 11, 2, 0.1089, 10, 2, 4, 8, 4.30, 2, 210.00, 12, 530.00, 9, 410.00, 0.0213, 5, 24, 148, 32, 0, 1, 2, 0, 21),
('100', '2026-05-30', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 88, 69, 19980.00, 84, 19120.00, 79, 18050.00, 227.62, 2, 77, 2, 1760.00, 910.00, 5, 6, 1, 0.0714, 5, 0, 1, 5, 4.70, 1, 90.00, 8, 280.00, 6, 220.00, 0.0146, 1, 17, 121, 18, 1, 0, 1, 0, 15),
('100', '2026-05-31', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 74, 58, 16890.00, 71, 16240.00, 68, 15410.00, 228.73, 1, 66, 1, 960.00, 620.00, 3, 5, 1, 0.0704, 4, 0, 1, 4, 4.80, 0, 0.00, 6, 180.00, 5, 150.00, 0.0111, 1, 14, 102, 15, 0, 0, 1, 0, 12),
('100', '2026-06-01', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 112, 87, 28640.00, 108, 27890.00, 101, 26310.00, 258.24, 6, 96, 5, 3520.00, 1990.00, 10, 12, 3, 0.1111, 12, 2, 5, 10, 4.20, 3, 330.00, 15, 720.00, 11, 520.00, 0.0258, 6, 28, 166, 38, 1, 1, 3, 1, 27),
('100', '2026-06-02', 'u100', 'POP商家', '品牌授权商家', '平台结算', '13800001000', '杭州云尚优选商贸有限公司', '91330100MA1000001X', 1, 1, '何林', '杭州市西湖区文三路88号', '杭州市余杭区仓前街道1号仓', '杭州市余杭区售后中心2号库', '招商银行杭州西湖支行', '6225888888881000', '对公账户', 0.0060, 8800.00, 91, 72, 21350.00, 87, 20580.00, 82, 19430.00, 236.55, 3, 80, 2, 2180.00, 1160.00, 6, 8, 2, 0.0920, 7, 1, 2, 6, 4.50, 1, 120.00, 10, 390.00, 7, 270.00, 0.0190, 2, 20, 132, 22, 0, 1, 1, 0, 16);

INSERT INTO dwm_trade_order_detail_di VALUES
('SO20260602001', '2026-06-02', 'O20260602001', 'buyer_001', 'buyer_001', '100', '待发货', 'SPU20260602001', '轻奢牛皮托特包 黑色', 'SKU20260602001', '轻奢牛皮托特包 黑色', 1, 399.00, '支付成功', '2026-06-02 09:18:00'),
('SO20260601001', '2026-06-01', 'O20260601001', 'buyer_002', 'buyer_002', '100', '已发货', 'SPU20260601001', '复古运动鞋 白蓝', 'SKU20260601001', '复古运动鞋 白蓝', 2, 698.00, '支付成功', '2026-06-01 14:35:00'),
('SO20260530001', '2026-05-30', 'O20260530001', 'buyer_003', 'buyer_003', '100', '交易成功', 'SPU20260530001', '夏季防晒衬衫 米色', 'SKU20260530001', '夏季防晒衬衫 米色', 1, 159.00, '支付成功', '2026-05-30 11:22:00');

INSERT INTO dwm_trade_refund_detail_di VALUES
('R20260601001', '2026-06-01', 'O20260601001', 'SO20260601001', '100', 'buyer_002', '待商家处理', '尺码不合适', '复古运动鞋 白蓝', 349.00, '2026-06-01 16:20:00'),
('R20260529001', '2026-05-29', 'O20260602001', 'SO20260602001', '100', 'buyer_001', '退款成功', '商品瑕疵', '轻奢牛皮托特包 黑色', 399.00, '2026-05-29 10:05:00');

INSERT INTO dwm_cs_ticket_detail_di VALUES
('T20260602001', '2026-06-02', '100', '买家反馈物流停滞', '处理中', '高', 0, 1, 4.00, '2026-06-02 10:12:00'),
('T20260601001', '2026-06-01', '100', '商品审核被拒原因咨询', '已关闭', '中', 1, 0, 4.50, '2026-06-01 13:40:00'),
('T20260529001', '2026-05-29', '100', '退款处理进度咨询', '已关闭', '中', 0, 0, 4.80, '2026-05-29 09:30:00');

INSERT INTO dwm_cs_repay_detail_df VALUES
('P20260601001', '2026-06-01', '100', 'O20260601006', 'SO20260601006', 80.00, '赔付成功', '保证金扣款', '2026-06-01 17:10:00');

INSERT INTO dwm_coupon_detail_di VALUES
('C20260602001', '2026-06-02', '100', 'buyer_001', '夏季满300减30券', 30.00, '已发放', '2026-06-02 08:00:00'),
('C20260601001', '2026-06-01', '100', 'buyer_004', '老客复购满200减20券', 20.00, '已核销', '2026-06-01 12:15:00');

INSERT INTO dwm_goods_detail_df VALUES
('SPU20260602001', '2026-06-02', '100', '轻奢牛皮托特包 黑色', '审核中', '商家提交审核', 0, '等待平台审核', '2026-06-02 09:05:00'),
('SPU20260601001', '2026-06-01', '100', '复古运动鞋 白蓝', '审核拒绝', '平台审核', 0, '主图存在夸大宣传文案，请修改后重新提交', '2026-06-01 15:20:00'),
('SPU20260530001', '2026-05-30', '100', '夏季防晒衬衫 米色', '已上架', '平台审核', 1, '审核通过', '2026-05-30 10:10:00');

INSERT INTO dwm_scm_detail_di VALUES
('IN20260601001', '2026-06-01', '100', '已入库', 'SPU20260530001', 'SKU20260530001', 60, 'WH-HZ-01', '质检通过', '正品', 'OUT20260601001', '2026-06-01 09:25:00');

INSERT INTO dwd_merchant_deposit_recharge_df VALUES
('D20260601001', '2026-06-01', '100', 'TRANS20260601001', 'CNY', 1000.00, '保证金补缴', '2026-06-01 11:00:00', '2026-06-01 11:01:00');

INSERT INTO dwd_merchant_appeal_detail_df VALUES
('A20260601001', '2026-06-01', '100', 'SPU20260601001', '复古运动鞋 白蓝', '申诉中', '商品审核申诉', '商家补充品牌授权书后发起申诉', '2026-06-01 18:20:00');
