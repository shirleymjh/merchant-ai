package com.yshopping.merchantai.model;

/**
 * 指标口径定义。
 *
 * <p>把用户自然语言命中的指标映射到 ads_merchant_profile 的字段、展示名和单位。</p>
 */
public class MetricDefinition {
    private final String column;
    private final String displayName;
    private final String unit;

    public MetricDefinition(String column, String displayName, String unit) {
        this.column = column;
        this.displayName = displayName;
        this.unit = unit;
    }

    public String getColumn() {
        return column;
    }

    public String getDisplayName() {
        return displayName;
    }

    public String getUnit() {
        return unit;
    }
}
