package com.yshopping.merchantai.model;

/**
 * 商家问题的业务分类。
 *
 * <p>分类名会展示到前端，也会写入 merchant_ai_answer.question_category_name。</p>
 */
public enum QuestionCategory {
    PLATFORM_RULE("平台商家规则"),
    TRADE("电商交易"),
    REFUND("电商退货"),
    CS_TICKET("电商客服工单"),
    COMPENSATION("电商理赔/赔付"),
    COUPON("电商优惠券"),
    GOODS("商品管理"),
    MERCHANT_OTHER("商家其他信息"),
    IDENTITY("身份信息"),
    SCM("供应链"),
    UNKNOWN("未知");

    private final String displayName;

    QuestionCategory(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
