package com.yshopping.merchantai.model;

/**
 * 用户输入的意图有效性。
 *
 * <p>GREETING 和 INVALID 不写入 merchant_ai_answer；VALID 会进入 Doris 查询和采纳写库流程。</p>
 */
public enum IntentType {
    GREETING,
    VALID,
    INVALID
}
