package com.yshopping.merchantai.model;

/**
 * 当前进线商家信息。
 *
 * <p>由 dim_merchant_df 识别得到，后续所有 Doris 查询都会基于该 merchant_id 限定数据范围。</p>
 */
public class MerchantInfo {
    private String merchantId;
    private String userId;
    private String merchantName;
    private String companyName;

    public MerchantInfo() {
    }

    public MerchantInfo(String merchantId, String userId, String merchantName, String companyName) {
        this.merchantId = merchantId;
        this.userId = userId;
        this.merchantName = merchantName;
        this.companyName = companyName;
    }

    public String getMerchantId() {
        return merchantId;
    }

    public void setMerchantId(String merchantId) {
        this.merchantId = merchantId;
    }

    public String getUserId() {
        return userId;
    }

    public void setUserId(String userId) {
        this.userId = userId;
    }

    public String getMerchantName() {
        return merchantName;
    }

    public void setMerchantName(String merchantName) {
        this.merchantName = merchantName;
    }

    public String getCompanyName() {
        return companyName;
    }

    public void setCompanyName(String companyName) {
        this.companyName = companyName;
    }
}
