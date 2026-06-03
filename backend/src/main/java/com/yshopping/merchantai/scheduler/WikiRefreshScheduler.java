package com.yshopping.merchantai.scheduler;

import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.repository.AnswerRepository;
import com.yshopping.merchantai.service.WikiMemoryService;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
/**
 * wiki 记忆自动压缩定时器。
 *
 * <p>每天凌晨把近期问答按分类压缩到 runtime/wiki，形成可复用的记忆闭环。</p>
 */
public class WikiRefreshScheduler {
    private static final Logger log = LoggerFactory.getLogger(WikiRefreshScheduler.class);
    private final AppProperties properties;
    private final AnswerRepository answerRepository;
    private final WikiMemoryService wikiMemoryService;

    public WikiRefreshScheduler(
            AppProperties properties,
            AnswerRepository answerRepository,
            WikiMemoryService wikiMemoryService) {
        this.properties = properties;
        this.answerRepository = answerRepository;
        this.wikiMemoryService = wikiMemoryService;
    }

    @Scheduled(cron = "0 30 3 * * *", zone = "Asia/Shanghai")
    public void refreshWiki() {
        Map<String, Path> outputs = wikiMemoryService.compressMerchantWiki(
                properties.getMerchantId(),
                Arrays.stream(QuestionCategory.values()).toList(),
                (merchantId, categoryName) -> answerRepository.recentAnswersByCategory(merchantId, categoryName, 200)
        );
        if (outputs.isEmpty()) {
            log.info("wiki 自动压缩完成，本次没有可更新的分类记忆。merchantId={}", properties.getMerchantId());
            return;
        }
        log.info("wiki 自动压缩完成，merchantId={}, outputs={}", properties.getMerchantId(), outputs);
    }
}
