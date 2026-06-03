package com.yshopping.merchantai.model;

/**
 * 平台规则问题的细分主题。
 *
 * <p>QuestionCategory 只表示一级业务域；RuleTopic 用来承载“商品上架规则”
 * 这类更具体的规则场景，避免规则类问题只能返回泛化话术。</p>
 */
public enum RuleTopic {
    GOODS_LISTING("商品上架规则"),
    GOODS_CATEGORY("商品类目规范"),
    GOODS_QUALIFICATION("商品资质要求"),
    GOODS_CONTENT("商品图文规范"),
    GOODS_REJECT("商品审核拒绝处理"),
    DELIVERY("交易发货规则"),
    REFUND("退货退款规则"),
    CS_TICKET("客服工单规则"),
    COMPENSATION("理赔赔付规则"),
    COUPON("优惠券规则"),
    DEPOSIT("保证金规则"),
    GENERAL("平台商家规则");

    private final String displayName;

    RuleTopic(String displayName) {
        this.displayName = displayName;
    }

    public String getDisplayName() {
        return displayName;
    }
}
