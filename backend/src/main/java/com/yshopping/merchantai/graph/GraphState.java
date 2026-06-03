package com.yshopping.merchantai.graph;

import com.yshopping.merchantai.model.MerchantInfo;
import com.yshopping.merchantai.model.QueryBundle;
import com.yshopping.merchantai.model.QueryPlan;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * LangGraph 全流程共享状态。
 *
 * <p>一次用户咨询从输入到输出只维护这一份状态，避免每个节点重复传递大量参数。</p>
 */
public class GraphState {
    /** 本次问答 id；前端采纳时用它关联待写入记录。 */
    private String id;
    /** 用户原始问题。 */
    private String question;
    /** 前端透传的商家 id；为空时使用默认商家 100。 */
    private String requestedMerchantId;
    /** 从 dim_merchant_df 识别出的商家信息。 */
    private MerchantInfo merchant;
    /** 多意图查询计划。 */
    private QueryPlan plan = new QueryPlan();
    /** 已加载的平台规则和历史 wiki 记忆。 */
    private String wiki = "";
    /** Doris 查询 SQL、用表和结果行。 */
    private QueryBundle queryBundle = new QueryBundle();
    /** 多意图下每个子问题的 Doris 查询结果。 */
    private List<QueryBundle> queryBundles = new ArrayList<>();
    /** 最终返回给商家的自然语言回答。 */
    private String answer = "";
    /** 是否属于有效业务问题；无效意图和寒暄不进入 Doris/写库。 */
    private boolean shouldPersist;
    /** 当前回答是否已写入 merchant_ai_answer。 */
    private boolean persisted;
    /** 猜你想问 3 条。 */
    private List<String> suggestions = new ArrayList<>();
    /** 前端“思考完成”展示的流程步骤。 */
    private List<String> thinkingSteps = new ArrayList<>();
    /** 历史问答记录，用于理解用户偏好和生成猜你想问。 */
    private List<Map<String, Object>> historyRows = new ArrayList<>();

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public String getRequestedMerchantId() {
        return requestedMerchantId;
    }

    public void setRequestedMerchantId(String requestedMerchantId) {
        this.requestedMerchantId = requestedMerchantId;
    }

    public MerchantInfo getMerchant() {
        return merchant;
    }

    public void setMerchant(MerchantInfo merchant) {
        this.merchant = merchant;
    }

    public QueryPlan getPlan() {
        return plan;
    }

    public void setPlan(QueryPlan plan) {
        this.plan = plan;
    }

    public String getWiki() {
        return wiki;
    }

    public void setWiki(String wiki) {
        this.wiki = wiki;
    }

    public QueryBundle getQueryBundle() {
        return queryBundle;
    }

    public void setQueryBundle(QueryBundle queryBundle) {
        this.queryBundle = queryBundle;
    }

    public List<QueryBundle> getQueryBundles() {
        return queryBundles;
    }

    public void setQueryBundles(List<QueryBundle> queryBundles) {
        this.queryBundles = queryBundles;
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public boolean isShouldPersist() {
        return shouldPersist;
    }

    public void setShouldPersist(boolean shouldPersist) {
        this.shouldPersist = shouldPersist;
    }

    public boolean isPersisted() {
        return persisted;
    }

    public void setPersisted(boolean persisted) {
        this.persisted = persisted;
    }

    public List<String> getSuggestions() {
        return suggestions;
    }

    public void setSuggestions(List<String> suggestions) {
        this.suggestions = suggestions;
    }

    public List<String> getThinkingSteps() {
        return thinkingSteps;
    }

    public List<Map<String, Object>> getHistoryRows() {
        return historyRows;
    }

    public void setHistoryRows(List<Map<String, Object>> historyRows) {
        this.historyRows = historyRows;
    }
}
