package com.yshopping.merchantai.dto;

/**
 * wiki 压缩请求体。
 *
 * <p>支持人工补充 Markdown，并结合历史问答沉淀为可复用 LLM 记忆。</p>
 */
public class WikiCompressRequest {
    private String categoryName;
    private String manualMarkdown = "";

    public String getCategoryName() {
        return categoryName;
    }

    public void setCategoryName(String categoryName) {
        this.categoryName = categoryName;
    }

    public String getManualMarkdown() {
        return manualMarkdown;
    }

    public void setManualMarkdown(String manualMarkdown) {
        this.manualMarkdown = manualMarkdown;
    }
}
