package com.yshopping.merchantai.dto;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 每日经营日报响应体。
 *
 * <p>用于前端经营数据卡片展示昨日核心指标和最近 7 日推导出的经营建议。</p>
 */
public class DailyReportResponse {
    private String merchantId;
    private String merchantName;
    private String date;
    private Map<String, Object> metrics;
    private List<String> suggestions = new ArrayList<>();

    public String getMerchantId() {
        return merchantId;
    }

    public void setMerchantId(String merchantId) {
        this.merchantId = merchantId;
    }

    public String getMerchantName() {
        return merchantName;
    }

    public void setMerchantName(String merchantName) {
        this.merchantName = merchantName;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public Map<String, Object> getMetrics() {
        return metrics;
    }

    public void setMetrics(Map<String, Object> metrics) {
        this.metrics = metrics;
    }

    public List<String> getSuggestions() {
        return suggestions;
    }

    public void setSuggestions(List<String> suggestions) {
        this.suggestions = suggestions;
    }
}
