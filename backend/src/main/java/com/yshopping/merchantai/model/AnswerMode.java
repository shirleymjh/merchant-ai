package com.yshopping.merchantai.model;

/**
 * 回答模式。
 *
 * <p>决定 LangGraph 后续节点走指标聚合、明细查询、规则文档、身份信息、
 * 普通寒暄还是无效意图分支。</p>
 */
public enum AnswerMode {
    METRIC,
    DETAIL,
    RULE,
    IDENTITY,
    CHAT,
    INVALID
}
