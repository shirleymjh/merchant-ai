package com.yshopping.merchantai.service;

import com.yshopping.merchantai.repository.AnswerRepository;
import java.time.LocalDateTime;
import org.springframework.stereotype.Service;

@Service
/**
 * 回答反馈服务。
 *
 * <p>问答生成时已写入 merchant_ai_answer，采纳时把同一条记录的 is_adopted 更新为 1；
 * 点赞/点踩先更新缓存，如果记录已存在则同步更新数据库字段。</p>
 */
public class FeedbackService {
    private final PendingAnswerStore pendingAnswerStore;
    private final AnswerRepository answerRepository;

    public FeedbackService(PendingAnswerStore pendingAnswerStore, AnswerRepository answerRepository) {
        this.pendingAnswerStore = pendingAnswerStore;
        this.answerRepository = answerRepository;
    }

    public boolean applyFeedback(String id, Boolean adopted, Boolean liked, Boolean disliked) {
        pendingAnswerStore.get(id).ifPresent(pending -> {
            if (Boolean.TRUE.equals(liked)) {
                pending.setLiked(true);
                pending.setDisliked(false);
            }
            if (Boolean.TRUE.equals(disliked)) {
                pending.setDisliked(true);
                pending.setLiked(false);
            }
        });

        if (Boolean.TRUE.equals(adopted)) {
            return pendingAnswerStore.get(id).map(pending -> {
                boolean persisted = answerRepository.insertAnswer(
                        pending.getId(),
                        pending.getQuestion(),
                        pending.getAnswer(),
                        pending.getMerchantId(),
                        pending.getMerchantName(),
                        pending.getCategoryName(),
                        pending.getDorisTables(),
                        pending.getSuggestedQuestions(),
                        pending.getCreateTime(),
                        LocalDateTime.now(),
                        true,
                        pending.isLiked() || Boolean.TRUE.equals(liked),
                        pending.isDisliked() || Boolean.TRUE.equals(disliked)
                );
                pendingAnswerStore.remove(id);
                return persisted;
            }).orElseGet(() -> {
                answerRepository.updateFeedback(id, true, liked, disliked);
                return answerRepository.isAvailable();
            });
        }

        if (answerRepository.exists(id)) {
            answerRepository.updateFeedback(id, adopted, liked, disliked);
            return answerRepository.isAvailable();
        }
        return false;
    }
}
