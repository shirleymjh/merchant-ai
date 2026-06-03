package com.yshopping.merchantai.service;

import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.QueryBundle;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.model.QuestionIntent;
import com.yshopping.merchantai.repository.DorisRepository;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * Doris 查询路由服务。
 *
 * <p>根据结构化意图选择 ads_merchant_profile 指标聚合、业务明细表查询或商家身份信息查询。
 * 最近 N 天指标会按 pt 分组，满足“超过 1 天按天汇总”的要求。</p>
 */
public class DorisQueryService {
    private static final ZoneId ZONE_ID = ZoneId.of("Asia/Shanghai");
    private final DorisRepository dorisRepository;

    public DorisQueryService(DorisRepository dorisRepository) {
        this.dorisRepository = dorisRepository;
    }

    public QueryBundle execute(String merchantId, QuestionIntent intent) {
        QueryBundle bundle = new QueryBundle();
        if (intent.getAnswerMode() == AnswerMode.METRIC) {
            queryMetric(merchantId, intent, bundle);
        } else if (intent.getAnswerMode() == AnswerMode.DETAIL) {
            queryDetail(merchantId, intent, bundle);
        } else if (intent.getAnswerMode() == AnswerMode.IDENTITY) {
            queryIdentity(merchantId, bundle);
        }
        return bundle;
    }

    public QueryBundle dailyReport(String merchantId) {
        QueryBundle bundle = new QueryBundle();
        LocalDate yesterday = LocalDate.now(ZONE_ID).minusDays(1);
        String sql = """
                SELECT
                  merchant_id,
                  pt,
                  order_gmv_amt_1d,
                  order_user_cnt_1d,
                  order_cnt_1d,
                  trade_success_order_cnt_1d,
                  return_cnt_1d,
                  refund_amt_1d,
                  goods_audit_reject_cnt_1d,
                  ship_timeout_order_cnt_1d,
                  avg_ticket_score_1d
                FROM ads_merchant_profile
                WHERE merchant_id = ? AND pt = ?
                LIMIT 1
                """;
        bundle.getTables().add("ads_merchant_profile");
        bundle.getSqls().add(sql);
        bundle.getRows().addAll(dorisRepository.query(sql, merchantId, yesterday));
        return bundle;
    }

    public List<Map<String, Object>> recentProfile(String merchantId, int days) {
        LocalDate end = LocalDate.now(ZONE_ID);
        LocalDate start = end.minusDays(Math.max(days, 1) - 1L);
        return dorisRepository.query(
                """
                SELECT pt, order_cnt_1d, order_gmv_amt_1d, return_cnt_1d, refund_amt_1d,
                       cs_ticket_cnt_1d, seller_repay_amt_1d, goods_audit_reject_cnt_1d,
                       ship_timeout_order_cnt_1d, avg_ticket_score_1d
                FROM ads_merchant_profile
                WHERE merchant_id = ? AND pt BETWEEN ? AND ?
                ORDER BY pt
                """,
                merchantId,
                start,
                end
        );
    }

    private void queryMetric(String merchantId, QuestionIntent intent, QueryBundle bundle) {
        String metricColumn = intent.getMetricColumn();
        if (!StringUtils.hasText(metricColumn)) {
            return;
        }
        LocalDate end = LocalDate.now(ZONE_ID);
        LocalDate start = end.minusDays(intent.getDays() - 1L);
        String sql = """
                SELECT ? AS metric_name, CAST(pt AS CHAR) AS pt, SUM(%s) AS value
                FROM ads_merchant_profile
                WHERE merchant_id = ? AND pt BETWEEN ? AND ?
                GROUP BY pt
                ORDER BY pt
                """.formatted(metricColumn);
        bundle.getTables().add("ads_merchant_profile");
        bundle.getSqls().add(sql);
        bundle.getRows().addAll(dorisRepository.query(sql, intent.getMetricName(), merchantId, start, end));
    }

    private void queryDetail(String merchantId, QuestionIntent intent, QueryBundle bundle) {
        DetailRoute route = detailRoute(intent);
        if (route == null) {
            return;
        }
        LocalDate end = LocalDate.now(ZONE_ID);
        LocalDate start = end.minusDays(intent.getDays() - 1L);
        bundle.getTables().add(route.table());
        bundle.getSqls().add(route.sql());
        bundle.getRows().addAll(dorisRepository.query(route.sql(), merchantId, start, end));
    }

    private void queryIdentity(String merchantId, QueryBundle bundle) {
        String sql = """
                SELECT user_id, merchant_id, merchant_type_name, brand_type_name, balance_type_name,
                       mobile, company_name, license_id, contact_name, business_address,
                       send_address, refnd_address, bank_name, bank_account, account_type_name,
                       ship_model_name, is_invoice, is_unconditional_refund,
                       init_deposit_amt, deposit_freeze, deposit_amt,
                       min_poundage, max_poundage, poundage_discount, CAST(pt AS CHAR) AS pt
                FROM dim_merchant_df
                WHERE merchant_id = ?
                ORDER BY pt DESC
                LIMIT 1
                """;
        bundle.getTables().add("dim_merchant_df");
        bundle.getSqls().add(sql);
        bundle.getRows().addAll(dorisRepository.query(sql, merchantId));
    }

    private DetailRoute detailRoute(QuestionIntent intent) {
        QuestionCategory category = intent.getCategory();
        String question = intent.getQuestion();
        return switch (category) {
            case TRADE -> new DetailRoute(
                    "dwm_trade_order_detail_di",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, order_id, sub_order_id, buyer_id, sub_order_status_name,
                           sku_name, sku_cnt, pay_amt, pay_status_name, sub_order_create_time
                    FROM dwm_trade_order_detail_di
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, sub_order_create_time DESC
                    """
            );
            case REFUND -> new DetailRoute(
                    "dwm_trade_refund_detail_di",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, refund_id, order_id, sub_order_id, refund_status_name,
                           refund_reason, sku_title, pay_amt, refund_create_time
                    FROM dwm_trade_refund_detail_di
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, refund_create_time DESC
                    """
            );
            case CS_TICKET -> new DetailRoute(
                    "dwm_cs_ticket_detail_di",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, ticket_id, ticket_title, ticket_status_name,
                           priority_name, is_reopen, is_reminder, ticket_score, ticket_create_time
                    FROM dwm_cs_ticket_detail_di
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, ticket_create_time DESC
                    """
            );
            case COMPENSATION -> new DetailRoute(
                    "dwm_cs_repay_detail_df",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, bill_id, order_id, sub_order_id, repay_amt,
                           repay_status_name, pay_way_name, create_time
                    FROM dwm_cs_repay_detail_df
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, create_time DESC
                    """
            );
            case COUPON -> new DetailRoute(
                    "dwm_coupon_detail_di",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, coupon_id, user_id, template_title, coupon_amt,
                           coupon_send_status_name, coupon_create_time
                    FROM dwm_coupon_detail_di
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, coupon_create_time DESC
                    """
            );
            case GOODS -> new DetailRoute(
                    "dwm_goods_detail_df",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, spu_id, spu_name, spu_status_name, audit_operate_type_name,
                           is_audit_pass, audit_remark, spu_apply_create_time
                    FROM dwm_goods_detail_df
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, spu_apply_create_time DESC
                    """
            );
            case MERCHANT_OTHER -> merchantOtherDetailRoute(question);
            case SCM -> new DetailRoute(
                    "dwm_scm_detail_di",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, inbound_id, inbound_status_name, spu_id, sku_id,
                           inbound_cnt, warehouse_id, check_status_name, identify_result_name, outbound_id
                    FROM dwm_scm_detail_di
                    WHERE seller_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, inbound_create_time DESC
                    """
            );
            default -> null;
        };
    }

    private DetailRoute merchantOtherDetailRoute(String question) {
        if (containsAny(question, "保证金") && containsAny(question, "充值", "补缴", "缴纳", "记录", "明细")) {
            return new DetailRoute(
                    "dwd_merchant_deposit_recharge_df",
                    """
                    SELECT CAST(pt AS CHAR) AS pt, deposit_recharge_id, trans_id, currency,
                           deposit_recharge_amt, remark, create_time, modify_time
                    FROM dwd_merchant_deposit_recharge_df
                    WHERE merchant_id = ?
                      AND pt BETWEEN ? AND ?
                    ORDER BY pt DESC, create_time DESC
                    """
            );
        }
        return new DetailRoute(
                "dwd_merchant_appeal_detail_df",
                """
                SELECT CAST(pt AS CHAR) AS pt, appeal_id, spu_id, spu_name, appeal_status_name,
                       apply_type_name, reason, create_time
                FROM dwd_merchant_appeal_detail_df
                WHERE CAST(merchant_id AS CHAR) = ?
                  AND pt BETWEEN ? AND ?
                ORDER BY pt DESC, create_time DESC
                """
        );
    }

    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (keyword != null && !keyword.isBlank() && text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private record DetailRoute(String table, String sql) {
    }
}
