package com.yshopping.merchantai.repository;

import com.yshopping.merchantai.config.AppProperties;
import jakarta.annotation.PostConstruct;
import java.nio.charset.StandardCharsets;
import java.sql.Timestamp;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.ClassPathResource;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;
import org.springframework.util.StreamUtils;

@Repository
/**
 * merchant_ai_answer 问答记录表访问层。
 *
 * <p>兼容独立 MySQL 和本地 Doris：优先尝试 MySQL DDL，失败后降级使用 Doris OLAP
 * 建表语句，方便本机直接联调采纳、点赞和点踩。</p>
 */
public class AnswerRepository {
    private static final Logger log = LoggerFactory.getLogger(AnswerRepository.class);
    private final JdbcTemplate jdbcTemplate;
    private final AppProperties properties;
    private volatile boolean available = true;

    public AnswerRepository(@Qualifier("answerJdbcTemplate") JdbcTemplate jdbcTemplate, AppProperties properties) {
        this.jdbcTemplate = jdbcTemplate;
        this.properties = properties;
    }

    @PostConstruct
    /**
     * 应用启动时初始化问答记录表。
     *
     * <p>如果 answer 数据源不可用，available 会置为 false，问答流程仍可运行，
     * 但采纳写库会被跳过并记录日志。</p>
     */
    public void initSchema() {
        try {
            ClassPathResource resource = new ClassPathResource("sql/merchant_ai_answer.sql");
            String sql = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
            jdbcTemplate.execute(sql);
        } catch (Exception e) {
            try {
                ClassPathResource resource = new ClassPathResource("sql/merchant_ai_answer_doris.sql");
                String sql = StreamUtils.copyToString(resource.getInputStream(), StandardCharsets.UTF_8);
                jdbcTemplate.execute(sql);
                available = true;
            } catch (Exception fallback) {
                available = false;
                log.warn("merchant_ai_answer 初始化失败，问答记录将暂时跳过写入。answer_url={}", properties.getDatasource().getAnswer().getUrl(), fallback);
            }
        }
    }

    public boolean isAvailable() {
        return available;
    }

    public boolean insertAnswer(
            String id,
            String question,
            String answer,
            String merchantId,
            String merchantName,
            String categoryName,
            String dorisTables,
            String suggestedQuestions,
            LocalDateTime createTime,
            LocalDateTime modifyTime
    ) {
        return insertAnswer(id, question, answer, merchantId, merchantName, categoryName, dorisTables,
                suggestedQuestions, createTime, modifyTime, false, false, false);
    }

    public boolean insertAnswer(
            String id,
            String question,
            String answer,
            String merchantId,
            String merchantName,
            String categoryName,
            String dorisTables,
            String suggestedQuestions,
            LocalDateTime createTime,
            LocalDateTime modifyTime,
            boolean adopted,
            boolean liked,
            boolean disliked
    ) {
        if (!available) {
            return false;
        }
        try {
            // Doris 不支持 MySQL 的 ON DUPLICATE KEY 语法，因此先查存在性再 insert/update。
            if (exists(id)) {
                jdbcTemplate.update(
                        """
                        UPDATE merchant_ai_answer
                        SET question = ?, answer = ?, is_adopted = ?, like_flag = ?, dislike_flag = ?,
                            merchant_id = ?, merchant_name = ?, question_category_name = ?,
                            doris_tables = ?, suggested_questions = ?, modify_time = ?
                        WHERE id = ?
                        """,
                        question,
                        answer,
                        adopted ? 1 : 0,
                        liked ? 1 : 0,
                        disliked ? 1 : 0,
                        merchantId,
                        merchantName,
                        categoryName,
                        dorisTables,
                        suggestedQuestions,
                        Timestamp.valueOf(modifyTime),
                        id
                );
            } else {
                jdbcTemplate.update(
                        """
                        INSERT INTO merchant_ai_answer
                        (id, question, answer, is_adopted, like_flag, dislike_flag, merchant_id, merchant_name,
                         question_category_name, doris_tables, suggested_questions, create_time, modify_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        id,
                        question,
                        answer,
                        adopted ? 1 : 0,
                        liked ? 1 : 0,
                        disliked ? 1 : 0,
                        merchantId,
                        merchantName,
                        categoryName,
                        dorisTables,
                        suggestedQuestions,
                        Timestamp.valueOf(createTime),
                        Timestamp.valueOf(modifyTime)
                );
            }
            return true;
        } catch (Exception e) {
            log.warn("写入 merchant_ai_answer 失败，id={}", id, e);
            return false;
        }
    }

    public boolean exists(String id) {
        if (!available) {
            return false;
        }
        try {
            Integer count = jdbcTemplate.queryForObject(
                    "SELECT COUNT(*) FROM merchant_ai_answer WHERE id = ?",
                    Integer.class,
                    id
            );
            return count != null && count > 0;
        } catch (Exception e) {
            return false;
        }
    }

    public void updateFeedback(String id, Boolean adopted, Boolean liked, Boolean disliked) {
        if (!available) {
            return;
        }
        try {
            // null 表示前端本次没有更新该字段，保留原值。
            Integer isAdopted = adopted == null ? null : (adopted ? 1 : 0);
            Integer likeFlag = liked == null ? null : (liked ? 1 : 0);
            Integer dislikeFlag = disliked == null ? null : (disliked ? 1 : 0);
            jdbcTemplate.update(
                    """
                    UPDATE merchant_ai_answer
                    SET is_adopted = COALESCE(?, is_adopted),
                        like_flag = COALESCE(?, like_flag),
                        dislike_flag = COALESCE(?, dislike_flag),
                        modify_time = ?
                    WHERE id = ?
                    """,
                    isAdopted,
                    likeFlag,
                    dislikeFlag,
                    Timestamp.valueOf(LocalDateTime.now()),
                    id
            );
        } catch (Exception e) {
            log.warn("更新问答反馈失败，id={}", id, e);
        }
    }

    public List<Map<String, Object>> recentAnswers(String merchantId, int limit) {
        if (!available) {
            return List.of();
        }
        try {
            return jdbcTemplate.queryForList(
                    """
                    SELECT question, answer, question_category_name, doris_tables, create_time
                    FROM merchant_ai_answer
                    WHERE merchant_id = ? AND is_adopted = 1
                    ORDER BY create_time DESC
                    LIMIT ?
                    """,
                    merchantId,
                    limit
            );
        } catch (Exception e) {
            log.warn("读取历史问答失败，merchantId={}", merchantId, e);
            return List.of();
        }
    }

    public List<Map<String, Object>> topCategoryQuestions(String merchantId, String categoryName, int limit) {
        if (!available) {
            return List.of();
        }
        try {
            return jdbcTemplate.queryForList(
                    """
                    SELECT question, COUNT(*) cnt
                    FROM merchant_ai_answer
                    WHERE merchant_id = ? AND question_category_name = ? AND is_adopted = 1
                    GROUP BY question
                    ORDER BY cnt DESC, MAX(create_time) DESC
                    LIMIT ?
                    """,
                    merchantId,
                    categoryName,
                    limit
            );
        } catch (Exception e) {
            return List.of();
        }
    }

    public List<Map<String, Object>> recentAnswersByCategory(String merchantId, String categoryName, int limit) {
        if (!available) {
            return List.of();
        }
        try {
            return jdbcTemplate.queryForList(
                    """
                    SELECT question, answer, question_category_name, doris_tables, create_time
                    FROM merchant_ai_answer
                    WHERE merchant_id = ? AND question_category_name LIKE ? AND is_adopted = 1
                    ORDER BY create_time DESC
                    LIMIT ?
                    """,
                    merchantId,
                    "%" + categoryName + "%",
                    limit
            );
        } catch (Exception e) {
            log.warn("按分类读取历史问答失败，merchantId={}, categoryName={}", merchantId, categoryName, e);
            return List.of();
        }
    }
}
