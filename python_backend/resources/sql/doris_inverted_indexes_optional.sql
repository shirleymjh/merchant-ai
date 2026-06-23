-- Optional Doris inverted index migration template.
-- Review table size, Doris version, build window, and query patterns before running.
-- The harness does not assume these indexes exist; accessHints keep invertedIndexes empty
-- until SHOW INDEX confirms real index definitions.

-- Order lookup and SPU analysis.
-- ALTER TABLE dwm_trade_order_detail_di ADD INDEX idx_order_id (order_id) USING INVERTED;
-- ALTER TABLE dwm_trade_order_detail_di ADD INDEX idx_sub_order_id (sub_order_id) USING INVERTED;
-- ALTER TABLE dwm_trade_order_detail_di ADD INDEX idx_spu_id (spu_id) USING INVERTED;
-- ALTER TABLE dwm_trade_order_detail_di ADD INDEX idx_spu_name (spu_name) USING INVERTED;

-- Refund lookup and refund goods analysis.
-- ALTER TABLE dwm_trade_refund_detail_di ADD INDEX idx_refund_id (refund_id) USING INVERTED;
-- ALTER TABLE dwm_trade_refund_detail_di ADD INDEX idx_refund_sub_order_id (sub_order_id) USING INVERTED;
-- ALTER TABLE dwm_trade_refund_detail_di ADD INDEX idx_refund_spu_name (spu_name) USING INVERTED;

-- Goods snapshot lookup.
-- ALTER TABLE dwm_goods_detail_df ADD INDEX idx_goods_spu_id (spu_id) USING INVERTED;
-- ALTER TABLE dwm_goods_detail_df ADD INDEX idx_goods_spu_name (spu_name) USING INVERTED;
-- ALTER TABLE dwm_goods_detail_df ADD INDEX idx_goods_audit_pass (is_audit_pass) USING INVERTED;

-- Ticket and compensation correlation.
-- ALTER TABLE dwm_cs_ticket_detail_di ADD INDEX idx_ticket_sub_order_id (sub_order_id) USING INVERTED;
-- ALTER TABLE dwm_cs_ticket_detail_di ADD INDEX idx_ticket_spu_id (spu_id) USING INVERTED;
-- ALTER TABLE dwm_cs_repay_detail_df ADD INDEX idx_repay_sub_order_id (sub_order_id) USING INVERTED;
-- ALTER TABLE dwm_cs_repay_detail_df ADD INDEX idx_repay_ticket_id (ticket_id) USING INVERTED;

-- Coupon and SCM follow-up.
-- ALTER TABLE dwm_coupon_detail_di ADD INDEX idx_coupon_id (coupon_id) USING INVERTED;
-- ALTER TABLE dwm_scm_detail_di ADD INDEX idx_scm_spu_id (spu_id) USING INVERTED;
-- ALTER TABLE dwm_scm_detail_di ADD INDEX idx_scm_spu_name (spu_name) USING INVERTED;
