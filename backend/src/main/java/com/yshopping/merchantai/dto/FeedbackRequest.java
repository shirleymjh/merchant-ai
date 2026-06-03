package com.yshopping.merchantai.dto;

/**
 * 回答反馈请求体。
 *
 * <p>采纳用于触发问答记录写入；点赞/点踩用于更新已采纳回答的反馈字段，
 * 或先缓存在待采纳回答里。</p>
 */
public class FeedbackRequest {
    private Boolean adopted;
    private Boolean liked;
    private Boolean disliked;

    public Boolean getAdopted() {
        return adopted;
    }

    public void setAdopted(Boolean adopted) {
        this.adopted = adopted;
    }

    public Boolean getLiked() {
        return liked;
    }

    public void setLiked(Boolean liked) {
        this.liked = liked;
    }

    public Boolean getDisliked() {
        return disliked;
    }

    public void setDisliked(Boolean disliked) {
        this.disliked = disliked;
    }
}
