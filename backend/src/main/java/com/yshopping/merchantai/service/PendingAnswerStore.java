package com.yshopping.merchantai.service;

import com.yshopping.merchantai.model.PendingAnswer;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import org.springframework.stereotype.Service;

@Service
/**
 * 待采纳回答内存缓存。
 *
 * <p>回答生成后先缓存 6 小时，避免用户未采纳的问题污染 merchant_ai_answer。</p>
 */
public class PendingAnswerStore {
    private static final Duration TTL = Duration.ofHours(6);
    private final ConcurrentHashMap<String, PendingAnswer> answers = new ConcurrentHashMap<>();

    public void put(PendingAnswer answer) {
        cleanup();
        answers.put(answer.getId(), answer);
    }

    public Optional<PendingAnswer> get(String id) {
        cleanup();
        return Optional.ofNullable(answers.get(id));
    }

    public void remove(String id) {
        answers.remove(id);
    }

    private void cleanup() {
        LocalDateTime deadline = LocalDateTime.now().minus(TTL);
        answers.entrySet().removeIf(entry -> entry.getValue().getCreateTime().isBefore(deadline));
    }
}
