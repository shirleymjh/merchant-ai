package com.yshopping.merchantai.dto;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * 商家 AI 助手返回给前端的回答结果。
 *
 * <p>除最终话术外，还包含问题分类、调用表、猜你想问、思考步骤和少量数据行，
 * 便于前端展示“思考完成”和引用数据来源。</p>
 */
public class ChatResponse {
    private String id;
    private String answer;
    private String categoryName;
    private boolean persisted;
    private List<String> dorisTables = new ArrayList<>();
    private List<String> suggestions = new ArrayList<>();
    private List<String> thinkingSteps = new ArrayList<>();
    private List<Map<String, Object>> dataRows = new ArrayList<>();

    public String getId() {
        return id;
    }

    public void setId(String id) {
        this.id = id;
    }

    public String getAnswer() {
        return answer;
    }

    public void setAnswer(String answer) {
        this.answer = answer;
    }

    public String getCategoryName() {
        return categoryName;
    }

    public void setCategoryName(String categoryName) {
        this.categoryName = categoryName;
    }

    public boolean isPersisted() {
        return persisted;
    }

    public void setPersisted(boolean persisted) {
        this.persisted = persisted;
    }

    public List<String> getDorisTables() {
        return dorisTables;
    }

    public void setDorisTables(List<String> dorisTables) {
        this.dorisTables = dorisTables;
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

    public void setThinkingSteps(List<String> thinkingSteps) {
        this.thinkingSteps = thinkingSteps;
    }

    public List<Map<String, Object>> getDataRows() {
        return dataRows;
    }

    public void setDataRows(List<Map<String, Object>> dataRows) {
        this.dataRows = dataRows;
    }
}
