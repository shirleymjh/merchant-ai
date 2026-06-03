package com.yshopping.merchantai.graph;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.dto.ChatResponse;
import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.IntentType;
import com.yshopping.merchantai.model.PendingAnswer;
import com.yshopping.merchantai.model.QueryBundle;
import com.yshopping.merchantai.model.QuestionIntent;
import com.yshopping.merchantai.repository.AnswerRepository;
import com.yshopping.merchantai.service.AnswerComposeService;
import com.yshopping.merchantai.service.DorisQueryService;
import com.yshopping.merchantai.service.IntentService;
import com.yshopping.merchantai.service.MerchantService;
import com.yshopping.merchantai.service.PendingAnswerStore;
import com.yshopping.merchantai.service.WikiMemoryService;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.UUID;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;

@Service
/**
 * yshopping 商家问答核心 LangGraph。
 *
 * <p>
 * 这里没有依赖外部图框架，而是用 GraphNode + GraphState 实现轻量状态图：
 * 用户输入会依次经过“商家识别、历史/wiki、意图识别、Doris 查询、回答整理、
 * 猜你想问、待采纳缓存”七个节点。
 * </p>
 *
 * <p>
 * 注意：有效业务回答生成后会立即写入 merchant_ai_answer，默认 is_adopted=0；
 * 用户点击“采纳”后更新为 1。寒暄和无效意图不会写库。
 * </p>
 */
public class MerchantQaLangGraph {
    private static final Logger log = LoggerFactory.getLogger(MerchantQaLangGraph.class);
    private final MerchantService merchantService;
    private final IntentService intentService;
    private final WikiMemoryService wikiMemoryService;
    private final DorisQueryService dorisQueryService;
    private final AnswerComposeService answerComposeService;
    private final AnswerRepository answerRepository;
    private final PendingAnswerStore pendingAnswerStore;
    private final ObjectMapper objectMapper;

    public MerchantQaLangGraph(
            // 负责识别当前商家是谁
            MerchantService merchantService,
            IntentService intentService,
            WikiMemoryService wikiMemoryService,
            DorisQueryService dorisQueryService,
            AnswerComposeService answerComposeService,
            AnswerRepository answerRepository,
            PendingAnswerStore pendingAnswerStore,
            ObjectMapper objectMapper) {
        this.merchantService = merchantService;
        this.intentService = intentService;
        this.wikiMemoryService = wikiMemoryService;
        this.dorisQueryService = dorisQueryService;
        this.answerComposeService = answerComposeService;
        this.answerRepository = answerRepository;
        this.pendingAnswerStore = pendingAnswerStore;
        this.objectMapper = objectMapper;
    }

    public ChatResponse run(String question, String merchantId) {
        GraphState state = new GraphState();
        state.setId("qa_" + UUID.randomUUID().toString().replace("-", ""));
        state.setQuestion(question == null ? "" : question.trim());
        state.setRequestedMerchantId(merchantId);

        // 核心图执行顺序：每个节点只读写 GraphState，保证流程可追踪、可扩展。
        state = loadMerchant().apply(state);
        state = loadHistoryAndWiki().apply(state);
        state = recognizeIntent().apply(state);
        state = executeDoris().apply(state);
        state = composeAnswer().apply(state);
        state = suggestQuestions().apply(state);
        state = cacheAnswer().apply(state);
        return toResponse(state);
    }

    private GraphNode loadMerchant() {
        return state -> {
            // 节点 1：识别当前商家。默认本地商家为 100，也支持前端透传 merchantId。
            state.setMerchant(merchantService.currentMerchant(state.getRequestedMerchantId()));
            state.getThinkingSteps().add("识别商家信息完成");
            return state;
        };
    }

    private GraphNode loadHistoryAndWiki() {
        return state -> {
            // 节点 2：读取历史问答和 wiki 记忆，辅助后续意图理解和话术生成。
            state.setHistoryRows(answerRepository.recentAnswers(state.getMerchant().getMerchantId(), 20));
            state.setWiki(wikiMemoryService.loadBaseWiki());
            state.getThinkingSteps().add("读取历史 LLM wiki 记忆完成");
            return state;
        };
    }

    private GraphNode recognizeIntent() {
        return state -> {
            // 节点 3：先识别多意图查询计划，再按分类精确加载 wiki。
            state.setPlan(intentService.recognizePlan(state.getQuestion(), state.getHistoryRows(), state.getWiki()));
            state.setWiki(wikiMemoryService.loadRelevantWiki(state.getPlan().categories()));
            state.setShouldPersist(state.getPlan().getIntents().stream()
                    .anyMatch(intent -> intent.getIntentType() == IntentType.VALID));
            long llmAnalyzedCount = state.getPlan().getIntents().stream().filter(QuestionIntent::isLlmAnalyzed).count();
            long llmRequestedCount = state.getPlan().getIntents().stream().filter(QuestionIntent::isLlmAnalysisRequested).count();
            if (llmAnalyzedCount > 0) {
                state.getThinkingSteps().add(
                        state.getPlan().getIntents().size() > 1
                                ? "复杂问题大模型意图分析完成，已拆成 %d 个子问题".formatted(state.getPlan().getIntents().size())
                                : "复杂问题大模型意图分析完成");
            } else if (llmRequestedCount > 0) {
                state.getThinkingSteps().add("复杂问题大模型未启用，已使用本地意图识别");
            } else {
                state.getThinkingSteps().add(
                        state.getPlan().getIntents().size() > 1
                                ? "意图识别完成，已拆成 %d 个子问题".formatted(state.getPlan().getIntents().size())
                                : "意图识别完成");
            }
            return state;
        };
    }

    private GraphNode executeDoris() {
        return state -> {
            // 节点 4：只有有效业务问题才读取 Doris；寒暄和无效问题直接跳过。
            if (!state.isShouldPersist()) {
                return state;
            }
            List<QueryBundle> queryBundles = new ArrayList<>();
            QueryBundle mergedBundle = new QueryBundle();
            boolean attemptedDoris = false;
            boolean success = false;
            for (QuestionIntent intent : state.getPlan().getIntents()) {
                if (intent.getIntentType() != IntentType.VALID) {
                    queryBundles.add(new QueryBundle());
                    continue;
                }
                if (intent.getAnswerMode() == AnswerMode.RULE) {
                    queryBundles.add(new QueryBundle());
                    continue;
                }
                try {
                    attemptedDoris = true;
                    QueryBundle bundle = dorisQueryService.execute(state.getMerchant().getMerchantId(), intent);
                    queryBundles.add(bundle);
                    mergeBundle(mergedBundle, bundle);
                    success = true;
                } catch (Exception e) {
                    log.warn("Doris 查询失败，category={}", intent.getCategory(), e);
                    queryBundles.add(new QueryBundle());
                }
            }
            state.setQueryBundles(queryBundles);
            state.setQueryBundle(mergedBundle);
            if (!attemptedDoris) {
                state.getThinkingSteps().add("规则知识库匹配完成");
            } else {
                state.getThinkingSteps().add(success ? "Doris 数据读取完成" : "Doris 数据读取失败，已进入降级回答");
            }
            return state;
        };
    }

    private GraphNode composeAnswer() {
        return state -> {
            // 节点 5：根据意图分支选择寒暄、无效提示或 LLM 数据型回答。
            if (state.getPlan().getIntents().size() <= 1) {
                state.setAnswer(composeSingleAnswer(state, primaryIntent(state), state.getQueryBundle()));
            } else {
                List<String> sections = new ArrayList<>();
                List<QuestionIntent> intents = state.getPlan().getIntents();
                for (int i = 0; i < intents.size(); i++) {
                    QuestionIntent intent = intents.get(i);
                    if (intent.getIntentType() != IntentType.VALID) {
                        continue;
                    }
                    QueryBundle bundle = i < state.getQueryBundles().size() ? state.getQueryBundles().get(i) : new QueryBundle();
                    sections.add("【%s】\n%s".formatted(
                            summarizeIntent(intent),
                            composeSingleAnswer(state, intent, bundle)));
                }
                state.setAnswer(sections.isEmpty()
                        ? composeSingleAnswer(state, primaryIntent(state), state.getQueryBundle())
                        : String.join("\n\n", sections));
            }
            if (state.isShouldPersist()) {
                state.setAnswer(answerComposeService.appendBusinessAdvice(
                        state.getAnswer(),
                        state.getPlan().getIntents(),
                        state.getQueryBundle()));
            }
            state.getThinkingSteps().add("回答整理完成");
            return state;
        };
    }

    private GraphNode suggestQuestions() {
        return state -> {
            // 节点 6：优先按当前问题分类读取 wiki 推荐问法，不足时再补历史高频问题。
            Set<String> suggestions = new LinkedHashSet<>();
            Set<String> normalizedSuggestions = new LinkedHashSet<>();
            String currentQuestion = normalizeSuggestion(state.getQuestion());
            for (QuestionIntent intent : state.getPlan().getIntents()) {
                String categoryName = displayCategory(intent);
                if (intent.getAnswerMode() == AnswerMode.RULE) {
                    addSuggestions(suggestions, normalizedSuggestions, currentQuestion,
                            answerComposeService.ruleSuggestions(intent.getRuleTopic()));
                } else {
                    addSuggestions(suggestions, normalizedSuggestions, currentQuestion,
                            wikiMemoryService.suggestedQuestions(intent.getCategory(), 3));
                }
                if (suggestions.size() >= 3) {
                    break;
                }
                addSuggestions(suggestions, normalizedSuggestions, currentQuestion,
                        answerComposeService.suggestions(
                                categoryName,
                                answerRepository.topCategoryQuestions(state.getMerchant().getMerchantId(), categoryName, 3)));
                if (suggestions.size() >= 3) {
                    break;
                }
            }
            state.setSuggestions(suggestions.stream().limit(3).toList());
            return state;
        };
    }

    private GraphNode cacheAnswer() {
        return state -> {
            // 节点 7：有效业务回答立即写库，默认未采纳；同时保留 Pending 方便后续采纳/点赞更新。
            LocalDateTime now = LocalDateTime.now();
            String suggestedJson = "[]";
            try {
                suggestedJson = objectMapper.writeValueAsString(state.getSuggestions());
            } catch (JsonProcessingException ignored) {
            }
            PendingAnswer pendingAnswer = new PendingAnswer(
                    state.getId(),
                    state.getQuestion(),
                    state.getAnswer(),
                    state.getMerchant().getMerchantId(),
                    state.getMerchant().getMerchantName(),
                    joinedCategories(state),
                    state.getQueryBundle().getTables().stream().distinct().reduce((left, right) -> left + "," + right).orElse(""),
                    suggestedJson,
                    now);
            if (state.isShouldPersist()) {
                pendingAnswerStore.put(pendingAnswer);
                boolean persisted = answerRepository.insertAnswer(
                        pendingAnswer.getId(),
                        pendingAnswer.getQuestion(),
                        pendingAnswer.getAnswer(),
                        pendingAnswer.getMerchantId(),
                        pendingAnswer.getMerchantName(),
                        pendingAnswer.getCategoryName(),
                        pendingAnswer.getDorisTables(),
                        pendingAnswer.getSuggestedQuestions(),
                        pendingAnswer.getCreateTime(),
                        now,
                        false,
                        false,
                        false);
                state.setPersisted(persisted);
                state.getThinkingSteps().add(persisted ? "已写入问答记录，等待采纳更新状态" : "问答记录写入失败，采纳时将重试");
            } else {
                state.setPersisted(false);
                state.getThinkingSteps().add("寒暄或无效意图不写入问答记录");
            }
            return state;
        };
    }

    private ChatResponse toResponse(GraphState state) {
        // 将图状态转换成前端响应体，隐藏后端内部对象，只暴露页面需要的数据。
        ChatResponse response = new ChatResponse();
        response.setId(state.getId());
        response.setAnswer(state.getAnswer());
        response.setCategoryName(joinedCategories(state));
        response.setPersisted(state.isPersisted());
        // 身份信息表只作为内部溯源写入 merchant_ai_answer，不在前端卡片展示表名或原始资料。
        if (state.getPlan().getIntents().stream().allMatch(intent -> intent.getAnswerMode() == AnswerMode.IDENTITY)) {
            response.setDorisTables(List.of());
            response.setDataRows(List.of());
        } else {
            response.setDorisTables(state.getQueryBundle().getTables().stream().distinct().toList());
            response.setDataRows(state.getQueryBundle().getRows());
        }
        response.setSuggestions(state.getSuggestions());
        response.setThinkingSteps(state.getThinkingSteps());
        return response;
    }

    private String composeSingleAnswer(GraphState state, QuestionIntent intent, QueryBundle bundle) {
        if (intent.getIntentType() == IntentType.GREETING) {
            return answerComposeService.greeting();
        }
        if (intent.getIntentType() == IntentType.INVALID) {
            return answerComposeService.invalid();
        }
        return answerComposeService.compose(
                state.getQuestion(),
                state.getMerchant(),
                intent,
                bundle,
                state.getWiki());
    }

    private void mergeBundle(QueryBundle target, QueryBundle source) {
        target.getTables().addAll(source.getTables());
        target.getSqls().addAll(source.getSqls());
        target.getRows().addAll(source.getRows());
    }

    private String joinedCategories(GraphState state) {
        return state.getPlan().getIntents().stream()
                .map(this::displayCategory)
                .distinct()
                .reduce((left, right) -> left + "," + right)
                .orElse(displayCategory(primaryIntent(state)));
    }

    private void addSuggestions(
            Set<String> suggestions,
            Set<String> normalizedSuggestions,
            String currentQuestion,
            List<String> candidates
    ) {
        if (candidates == null) {
            return;
        }
        for (String candidate : candidates) {
            String normalized = normalizeSuggestion(candidate);
            if (normalized.isBlank() || normalized.equals(currentQuestion) || !normalizedSuggestions.add(normalized)) {
                continue;
            }
            suggestions.add(candidate.trim());
            if (suggestions.size() >= 3) {
                return;
            }
        }
    }

    private String normalizeSuggestion(String question) {
        if (question == null) {
            return "";
        }
        return question
                .replaceAll("[\\s　?？!！。.,，;；:：、\"'“”‘’（）()【】\\[\\]]+", "")
                .trim();
    }

    private String displayCategory(QuestionIntent intent) {
        if (intent.getAnswerMode() == AnswerMode.RULE) {
            return intent.getRuleTopic().getDisplayName();
        }
        return intent.getCategory().getDisplayName();
    }

    private QuestionIntent primaryIntent(GraphState state) {
        return state.getPlan().primaryIntent();
    }

    private String summarizeIntent(QuestionIntent intent) {
        if (intent.getIntentType() == IntentType.GREETING) {
            return "寒暄";
        }
        if (intent.getIntentType() == IntentType.INVALID) {
            return "人工咨询";
        }
        if (intent.getAnswerMode() == AnswerMode.IDENTITY && !intent.getIdentityName().isBlank()) {
            return intent.getIdentityName();
        }
        if (intent.getAnswerMode() == AnswerMode.METRIC && !intent.getMetricName().isBlank()) {
            return intent.getCategory().getDisplayName() + "-" + intent.getMetricName();
        }
        return displayCategory(intent);
    }
}
