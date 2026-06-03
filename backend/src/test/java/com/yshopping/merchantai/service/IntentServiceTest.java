package com.yshopping.merchantai.service;

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.QueryPlan;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.model.QuestionIntent;
import com.yshopping.merchantai.model.RuleTopic;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class IntentServiceTest {
    private IntentService intentService;

    @BeforeEach
    void setUp() {
        ObjectMapper objectMapper = new ObjectMapper();
        LlmClient llmClient = new LlmClient(new AppProperties(), objectMapper);
        intentService = new IntentService(new LlmIntentAnalysisService(llmClient, objectMapper));
    }

    @Test
    void recognizesGoodsQualificationAsRule() {
        QuestionIntent intent = intentService.recognizePlan("上架商品需要哪些资质", null, "").primaryIntent();

        assertThat(intent.getCategory()).isEqualTo(QuestionCategory.PLATFORM_RULE);
        assertThat(intent.getAnswerMode()).isEqualTo(AnswerMode.RULE);
        assertThat(intent.getRuleTopic()).isEqualTo(RuleTopic.GOODS_QUALIFICATION);
    }

    @Test
    void recognizesShortCategoryFollowUpAsRule() {
        QuestionIntent intent = intentService.recognizePlan("类目规范呢", null, "").primaryIntent();

        assertThat(intent.getCategory()).isEqualTo(QuestionCategory.PLATFORM_RULE);
        assertThat(intent.getAnswerMode()).isEqualTo(AnswerMode.RULE);
        assertThat(intent.getRuleTopic()).isEqualTo(RuleTopic.GOODS_CATEGORY);
    }

    @Test
    void keepsGoodsAuditCountAsDataQuery() {
        QuestionIntent intent = intentService.recognizePlan("昨天商品审核拒绝量是多少", null, "").primaryIntent();

        assertThat(intent.getCategory()).isEqualTo(QuestionCategory.GOODS);
        assertThat(intent.getAnswerMode()).isEqualTo(AnswerMode.METRIC);
        assertThat(intent.getMetricColumn()).isEqualTo("goods_audit_reject_cnt_1d");
    }

    @Test
    void inheritsGlobalTimeRangeForSplitBusinessQuestions() {
        QueryPlan plan = intentService.recognizePlan("我想看最近7天工单量和退货数量", null, "");

        assertThat(plan.getIntents()).hasSize(2);
        assertThat(plan.getIntents()).allSatisfy(intent -> assertThat(intent.getDays()).isEqualTo(7));
        assertThat(plan.getIntents())
                .extracting(QuestionIntent::getCategory)
                .containsExactly(QuestionCategory.CS_TICKET, QuestionCategory.REFUND);
    }
}
