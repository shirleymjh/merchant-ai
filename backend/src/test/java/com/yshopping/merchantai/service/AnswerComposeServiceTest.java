package com.yshopping.merchantai.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.RuleTopic;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class AnswerComposeServiceTest {
    private AnswerComposeService answerComposeService;

    @BeforeEach
    void setUp() {
        ObjectMapper objectMapper = new ObjectMapper();
        answerComposeService = new AnswerComposeService(new LlmClient(new AppProperties(), objectMapper), objectMapper);
    }

    @Test
    void returnsRuleTopicSuggestionsForGoodsContent() {
        assertThat(answerComposeService.ruleSuggestions(RuleTopic.GOODS_CONTENT))
                .containsExactly(
                        "商品主图有哪些要求？",
                        "商品详情页描述怎么写？",
                        "图文不符会导致审核被拒吗？");
    }
}
