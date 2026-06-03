package com.yshopping.merchantai.service;

import com.yshopping.merchantai.dto.DailyReportResponse;
import com.yshopping.merchantai.model.MerchantInfo;
import com.yshopping.merchantai.model.QueryBundle;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
/**
 * 每日经营日报服务。
 *
 * <p>读取 ads_merchant_profile 昨日核心指标，并结合近 7 日画像生成两条经营建议。</p>
 */
public class DailyReportService {
    private static final ZoneId ZONE_ID = ZoneId.of("Asia/Shanghai");
    private final MerchantService merchantService;
    private final DorisQueryService dorisQueryService;

    public DailyReportService(MerchantService merchantService, DorisQueryService dorisQueryService) {
        this.merchantService = merchantService;
        this.dorisQueryService = dorisQueryService;
    }

    public DailyReportResponse report(String merchantId) {
        MerchantInfo merchant = merchantService.currentMerchant(merchantId);
        QueryBundle bundle = dorisQueryService.dailyReport(merchant.getMerchantId());
        Map<String, Object> row = bundle.getRows().isEmpty() ? Map.of() : bundle.getRows().get(0);
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("昨日总gmv金额", row.getOrDefault("order_gmv_amt_1d", 0));
        metrics.put("昨日下单用户量", row.getOrDefault("order_user_cnt_1d", 0));
        metrics.put("昨日总订单量", row.getOrDefault("order_cnt_1d", 0));
        metrics.put("昨日交易成功订单量", row.getOrDefault("trade_success_order_cnt_1d", 0));
        metrics.put("昨日退货量", row.getOrDefault("return_cnt_1d", 0));
        metrics.put("昨日退款金额", row.getOrDefault("refund_amt_1d", 0));

        DailyReportResponse response = new DailyReportResponse();
        response.setMerchantId(merchant.getMerchantId());
        response.setMerchantName(merchant.getMerchantName());
        response.setDate(LocalDate.now(ZONE_ID).minusDays(1).toString());
        response.setMetrics(metrics);
        response.setSuggestions(buildSuggestions(merchant.getMerchantId()));
        return response;
    }

    private List<String> buildSuggestions(String merchantId) {
        List<Map<String, Object>> rows = dorisQueryService.recentProfile(merchantId, 7);
        if (rows.isEmpty()) {
            return List.of("暂无近 7 日经营数据，建议保持商品供给和客服响应稳定。", "可以先补齐商品、保证金和商家资料，提升平台经营基础。");
        }
        double refundAmt = sum(rows, "refund_amt_1d");
        double orderCnt = sum(rows, "order_cnt_1d");
        double ticketCnt = sum(rows, "cs_ticket_cnt_1d");
        double rejectCnt = sum(rows, "goods_audit_reject_cnt_1d");
        String first = refundAmt > 0
                ? "近 7 日存在退款金额，建议优先查看退货退款明细，定位高频原因并优化发货/售后说明。"
                : "近 7 日退款压力较低，可以继续保持履约和售后响应稳定。";
        String second;
        if (ticketCnt > orderCnt * 0.2) {
            second = "客服工单相对订单量偏高，建议排查催单、物流和商品说明类问题。";
        } else if (rejectCnt > 0) {
            second = "存在商品审核拒绝记录，建议复查商品图片、品牌资质和类目填写。";
        } else {
            second = "建议继续关注 GMV、交易成功订单量和优惠使用效果，挑选转化较好的商品加大运营。";
        }
        return List.of(first, second);
    }

    private double sum(List<Map<String, Object>> rows, String key) {
        return rows.stream()
                .map(row -> row.get(key))
                .mapToDouble(value -> value instanceof Number number ? number.doubleValue() : 0D)
                .sum();
    }
}
