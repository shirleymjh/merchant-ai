package com.yshopping.merchantai.controller;

import com.yshopping.merchantai.dto.ChatRequest;
import com.yshopping.merchantai.dto.ChatResponse;
import com.yshopping.merchantai.dto.DailyReportResponse;
import com.yshopping.merchantai.dto.FeedbackRequest;
import com.yshopping.merchantai.dto.WikiCompressRequest;
import com.yshopping.merchantai.graph.MerchantQaLangGraph;
import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.repository.AnswerRepository;
import com.yshopping.merchantai.service.DailyReportService;
import com.yshopping.merchantai.service.FeedbackService;
import com.yshopping.merchantai.service.WikiMemoryService;
import jakarta.validation.Valid;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import org.springframework.util.StringUtils;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api")
/**
 * 商家 AI 助手 API 控制器。
 *
 * <p>对前端暴露问答、采纳/点赞/点踩反馈、每日经营日报以及 wiki 压缩接口。</p>
 */
public class ChatController {
    private final MerchantQaLangGraph merchantQaLangGraph;
    private final AppProperties properties;
    private final AnswerRepository answerRepository;
    private final FeedbackService feedbackService;
    private final WikiMemoryService wikiMemoryService;
    private final DailyReportService dailyReportService;

    public ChatController(
            MerchantQaLangGraph merchantQaLangGraph,
            AppProperties properties,
            AnswerRepository answerRepository,
            FeedbackService feedbackService,
            WikiMemoryService wikiMemoryService,
            DailyReportService dailyReportService
    ) {
        this.merchantQaLangGraph = merchantQaLangGraph;
        this.properties = properties;
        this.answerRepository = answerRepository;
        this.feedbackService = feedbackService;
        this.wikiMemoryService = wikiMemoryService;
        this.dailyReportService = dailyReportService;
    }

    @PostMapping("/chat")
    public ChatResponse chat(@Valid @RequestBody ChatRequest request) {
        return merchantQaLangGraph.run(request.getMessage(), request.getMerchantId());
    }

    @PostMapping("/answers/{id}/feedback")
    public Map<String, Object> feedback(@PathVariable String id, @RequestBody FeedbackRequest request) {
        boolean persisted = feedbackService.applyFeedback(id, request.getAdopted(), request.getLiked(), request.getDisliked());
        return Map.of("success", true, "persisted", persisted);
    }

    @GetMapping("/daily-report")
    public DailyReportResponse dailyReport(@RequestParam(required = false) String merchantId) {
        return dailyReportService.report(merchantId);
    }

    @PostMapping("/wiki/compress")
    public Map<String, Object> compressWiki(@RequestBody WikiCompressRequest request) {
        if (!StringUtils.hasText(request.getCategoryName())) {
            Map<String, Path> outputs = wikiMemoryService.compressMerchantWiki(
                    properties.getMerchantId(),
                    Arrays.stream(QuestionCategory.values()).toList(),
                    (merchantId, categoryName) -> answerRepository.recentAnswersByCategory(merchantId, categoryName, 200)
            );
            return Map.of("success", true, "paths", outputs);
        }
        Path path = wikiMemoryService.compressToWiki(
                request.getCategoryName(),
                answerRepository.recentAnswersByCategory(properties.getMerchantId(), request.getCategoryName(), 200),
                request.getManualMarkdown()
        );
        return Map.of("success", true, "path", path.toAbsolutePath().toString());
    }
}
