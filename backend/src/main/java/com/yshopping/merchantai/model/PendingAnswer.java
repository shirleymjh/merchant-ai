package com.yshopping.merchantai.model;

import java.time.LocalDateTime;

/**
 * 待采纳回答缓存对象。
 *
 * <p>对话完成后先放入内存缓存；用户点击“采纳”时再把这里的信息写入 merchant_ai_answer。</p>
 */
public class PendingAnswer {
    private final String id;
    private final String question;
    private final String answer;
    private final String merchantId;
    private final String merchantName;
    private final String categoryName;
    private final String dorisTables;
    private final String suggestedQuestions;
    private final LocalDateTime createTime;
    private boolean liked;
    private boolean disliked;

    public PendingAnswer(
            String id,
            String question,
            String answer,
            String merchantId,
            String merchantName,
            String categoryName,
            String dorisTables,
            String suggestedQuestions,
            LocalDateTime createTime
    ) {
        this.id = id;
        this.question = question;
        this.answer = answer;
        this.merchantId = merchantId;
        this.merchantName = merchantName;
        this.categoryName = categoryName;
        this.dorisTables = dorisTables;
        this.suggestedQuestions = suggestedQuestions;
        this.createTime = createTime;
    }

    public String getId() {
        return id;
    }

    public String getQuestion() {
        return question;
    }

    public String getAnswer() {
        return answer;
    }

    public String getMerchantId() {
        return merchantId;
    }

    public String getMerchantName() {
        return merchantName;
    }

    public String getCategoryName() {
        return categoryName;
    }

    public String getDorisTables() {
        return dorisTables;
    }

    public String getSuggestedQuestions() {
        return suggestedQuestions;
    }

    public LocalDateTime getCreateTime() {
        return createTime;
    }

    public boolean isLiked() {
        return liked;
    }

    public void setLiked(boolean liked) {
        this.liked = liked;
    }

    public boolean isDisliked() {
        return disliked;
    }

    public void setDisliked(boolean disliked) {
        this.disliked = disliked;
    }
}
