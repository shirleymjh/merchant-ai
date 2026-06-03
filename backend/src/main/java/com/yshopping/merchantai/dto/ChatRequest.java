package com.yshopping.merchantai.dto;

import jakarta.validation.constraints.NotBlank;

/**
 * 前端发起一次商家问答时提交的请求体。
 */
public class ChatRequest {
    @NotBlank
    private String message;
    private String merchantId;

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getMerchantId() {
        return merchantId;
    }

    public void setMerchantId(String merchantId) {
        this.merchantId = merchantId;
    }
}
