package com.yshopping.merchantai.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.IntentType;
import com.yshopping.merchantai.model.MetricDefinition;
import com.yshopping.merchantai.model.QueryPlan;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.model.QuestionIntent;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * 复杂问题 LLM 意图增强服务。
 *
 * <p>商家 AI 助手的核心目标，是把原本依赖人工客服流转的商家咨询，转成“自动理解、
 * 自动判断、自动回复”的结构化服务流程。规则意图识别适合高频、稳定问题；当问题较复杂、
 * 跨多个业务域或关键词无法覆盖时，本服务调用大模型把自然语言解析为结构化 JSON。</p>
 *
 * <p>为了避免模型直接决定 SQL，这里只接受白名单内的业务分类、回答模式、指标字段和
 * 身份字段，后续 Doris 查询仍由 DorisQueryService 使用固定模板生成。这样既能覆盖物流、
 * 商品、退款、费率、工单等高频商家问题，也能把回答过程约束在可审计的业务边界内。</p>
 */
public class LlmIntentAnalysisService {
    private static final int MAX_WIKI_CHARS = 4000;
    private static final int MAX_HISTORY_ROWS = 8;

    private static final Map<String, MetricDefinition> METRIC_COLUMNS = Map.ofEntries(
            Map.entry("order_gmv_amt_1d", new MetricDefinition("order_gmv_amt_1d", "总 GMV 金额", "元")),
            Map.entry("order_user_cnt_1d", new MetricDefinition("order_user_cnt_1d", "下单用户量", "人")),
            Map.entry("order_cnt_1d", new MetricDefinition("order_cnt_1d", "总订单量", "单")),
            Map.entry("trade_success_order_cnt_1d", new MetricDefinition("trade_success_order_cnt_1d", "交易成功订单量", "单")),
            Map.entry("trade_success_gmv_amt_1d", new MetricDefinition("trade_success_gmv_amt_1d", "交易成功 GMV 金额", "元")),
            Map.entry("avg_pay_order_amt_1d", new MetricDefinition("avg_pay_order_amt_1d", "支付成功客单价", "元")),
            Map.entry("ship_timeout_order_cnt_1d", new MetricDefinition("ship_timeout_order_cnt_1d", "发货超时订单量", "单")),
            Map.entry("refund_amt_1d", new MetricDefinition("refund_amt_1d", "退款金额", "元")),
            Map.entry("return_success_amt_1d", new MetricDefinition("return_success_amt_1d", "退货成功金额", "元")),
            Map.entry("return_success_cnt_1d", new MetricDefinition("return_success_cnt_1d", "退货成功量", "单")),
            Map.entry("return_cnt_1d", new MetricDefinition("return_cnt_1d", "退货量", "单")),
            Map.entry("direct_refund_cnt_1d", new MetricDefinition("direct_refund_cnt_1d", "直接退款量", "单")),
            Map.entry("refund_rate_1d", new MetricDefinition("refund_rate_1d", "退货量占支付订单量比例", "%")),
            Map.entry("cs_ticket_cnt_1d", new MetricDefinition("cs_ticket_cnt_1d", "咨询工单量", "个")),
            Map.entry("ticket_reopen_cnt_1d", new MetricDefinition("ticket_reopen_cnt_1d", "工单二次开启量", "个")),
            Map.entry("ticket_reminder_cnt_1d", new MetricDefinition("ticket_reminder_cnt_1d", "催单工单量", "个")),
            Map.entry("ticket_close_cnt_1d", new MetricDefinition("ticket_close_cnt_1d", "关闭工单量", "个")),
            Map.entry("avg_ticket_score_1d", new MetricDefinition("avg_ticket_score_1d", "平均工单评价分", "分")),
            Map.entry("seller_repay_amt_1d", new MetricDefinition("seller_repay_amt_1d", "卖家赔付金额", "元")),
            Map.entry("seller_repay_order_cnt_1d", new MetricDefinition("seller_repay_order_cnt_1d", "卖家赔付订单数", "单")),
            Map.entry("pay_success_discount_order_cnt_1d", new MetricDefinition("pay_success_discount_order_cnt_1d", "支付成功优惠单量", "单")),
            Map.entry("pay_success_discount_amt_1d", new MetricDefinition("pay_success_discount_amt_1d", "支付成功优惠金额", "元")),
            Map.entry("trade_success_discount_order_cnt_1d", new MetricDefinition("trade_success_discount_order_cnt_1d", "交易成功优惠单量", "单")),
            Map.entry("trade_success_discount_amt_1d", new MetricDefinition("trade_success_discount_amt_1d", "交易成功优惠金额", "元")),
            Map.entry("pay_discount_rate_1d", new MetricDefinition("pay_discount_rate_1d", "支付成功优惠金额占支付 GMV 比例", "%")),
            Map.entry("goods_audit_reject_cnt_1d", new MetricDefinition("goods_audit_reject_cnt_1d", "商品审核拒绝量", "个")),
            Map.entry("goods_audit_pass_cnt_1d", new MetricDefinition("goods_audit_pass_cnt_1d", "商品审核通过量", "个")),
            Map.entry("goods_online_cnt_1d", new MetricDefinition("goods_online_cnt_1d", "上架商品量", "个")),
            Map.entry("goods_apply_cnt_1d", new MetricDefinition("goods_apply_cnt_1d", "商品申请量", "个")),
            Map.entry("deposit_pay_cnt_1d", new MetricDefinition("deposit_pay_cnt_1d", "缴纳保证金次数", "次")),
            Map.entry("appeal_success_cnt_1d", new MetricDefinition("appeal_success_cnt_1d", "申诉成功次数", "次")),
            Map.entry("appeal_cnt_1d", new MetricDefinition("appeal_cnt_1d", "申诉次数", "次")),
            Map.entry("punish_cnt_1d", new MetricDefinition("punish_cnt_1d", "处罚次数", "次")),
            Map.entry("scm_performance_cnt_1d", new MetricDefinition("scm_performance_cnt_1d", "供应链履约量", "单"))
    );

    private static final Map<QuestionCategory, MetricDefinition> DEFAULT_METRICS = Map.of(
            QuestionCategory.TRADE, new MetricDefinition("order_cnt_1d", "总订单量", "单"),
            QuestionCategory.REFUND, new MetricDefinition("return_cnt_1d", "退货量", "单"),
            QuestionCategory.CS_TICKET, new MetricDefinition("cs_ticket_cnt_1d", "咨询工单量", "个"),
            QuestionCategory.COMPENSATION, new MetricDefinition("seller_repay_amt_1d", "卖家赔付金额", "元"),
            QuestionCategory.COUPON, new MetricDefinition("trade_success_discount_amt_1d", "交易成功优惠金额", "元"),
            QuestionCategory.GOODS, new MetricDefinition("goods_apply_cnt_1d", "商品申请量", "个"),
            QuestionCategory.MERCHANT_OTHER, new MetricDefinition("appeal_cnt_1d", "申诉次数", "次"),
            QuestionCategory.SCM, new MetricDefinition("scm_performance_cnt_1d", "供应链履约量", "单")
    );

    private static final Map<String, String> IDENTITY_COLUMNS = Map.ofEntries(
            Map.entry("company_name", "公司名称"),
            Map.entry("mobile", "商家手机号"),
            Map.entry("contact_name", "联系人姓名"),
            Map.entry("license_id", "营业执照编号"),
            Map.entry("send_address", "发货地址"),
            Map.entry("refnd_address", "退货地址"),
            Map.entry("business_address", "经营地址"),
            Map.entry("bank_name", "开户行"),
            Map.entry("bank_account", "银行账号"),
            Map.entry("merchant_type_name", "商户类型"),
            Map.entry("brand_type_name", "资质类型"),
            Map.entry("balance_type_name", "结算类型"),
            Map.entry("ship_model_name", "发货模式"),
            Map.entry("account_type_name", "账户类型"),
            Map.entry("is_invoice", "开具发票"),
            Map.entry("is_unconditional_refund", "七天无理由退货"),
            Map.entry("init_deposit_amt", "入驻初始保证金"),
            Map.entry("deposit_freeze", "冻结保证金"),
            Map.entry("deposit_amt", "保证金余额"),
            Map.entry("min_poundage", "手续费下限"),
            Map.entry("max_poundage", "手续费上限"),
            Map.entry("poundage_discount", "费率折扣")
    );

    private final LlmClient llmClient;
    private final ObjectMapper objectMapper;

    public LlmIntentAnalysisService(LlmClient llmClient, ObjectMapper objectMapper) {
        this.llmClient = llmClient;
        this.objectMapper = objectMapper;
    }

    public boolean isAvailable() {
        return llmClient.isConfigured();
    }

    public QuestionIntent refine(String question, QuestionIntent baseIntent, List<Map<String, Object>> historyRows, String wiki) {
        QuestionIntent fallbackIntent = copy(baseIntent);
        fallbackIntent.setLlmAnalysisRequested(true);
        if (!llmClient.isConfigured()) {
            fallbackIntent.setAnalysisNote("复杂问题需要大模型分析，但当前未配置 YSHOPPING_LLM_API_KEY，已使用规则识别结果");
            return fallbackIntent;
        }

        String response = llmClient.chat(systemPrompt(), buildPrompt(question, baseIntent, historyRows, wiki), "");
        if (!StringUtils.hasText(response)) {
            fallbackIntent.setAnalysisNote("大模型未返回可用意图，已使用规则识别结果");
            return fallbackIntent;
        }

        try {
            QuestionIntent refined = applyJsonIntent(question, baseIntent, extractJson(response));
            refined.setLlmAnalysisRequested(true);
            refined.setLlmAnalyzed(true);
            refined.setAnalysisSource("LLM");
            refined.setAnalysisNote("复杂问题已由大模型完成意图分析");
            return refined;
        } catch (Exception e) {
            fallbackIntent.setAnalysisNote("大模型意图 JSON 解析失败，已使用规则识别结果");
            return fallbackIntent;
        }
    }

    public QueryPlan refinePlan(String question, QueryPlan fallbackPlan, List<Map<String, Object>> historyRows, String wiki) {
        QueryPlan fallback = copyPlan(fallbackPlan);
        markPlanAsRequested(fallback);
        if (!llmClient.isConfigured()) {
            return fallback;
        }

        String response = llmClient.chat(planSystemPrompt(), buildPlanPrompt(question, fallbackPlan, historyRows, wiki), "");
        if (!StringUtils.hasText(response)) {
            return fallback;
        }

        try {
            QueryPlan refined = applyPlanJson(question, extractJson(response));
            refined = sanitizeRefinedPlan(refined, fallback);
            markPlanAsAnalyzed(refined);
            return refined;
        } catch (Exception e) {
            return fallback;
        }
    }

    private String systemPrompt() {
        return """
                你是 yshopping 商家 AI 助手的意图识别器。
                你的任务是帮助平台把商家原本需要提交工单的人工作业，尽量转成可自动处理的智能服务流程。
                你只负责把用户问题解析成白名单内的结构化意图，不要编写 SQL，不要回答用户。
                重点识别物流单号关联订单、商品上架、退款处理、费率规则、平台操作指南、客服工单进度等高频场景。
                如果问题仍属于这些业务域，请优先输出可执行意图；只有明显超出支持范围时才输出 INVALID。
                必须只输出 JSON 对象，不要输出 Markdown。
                """;
    }

    private String planSystemPrompt() {
        return """
                你是 yshopping 商家 AI 助手的多意图查询规划器。
                你要先理解用户问题，再把它拆成 1 到 4 个可执行的子意图。
                规划目标是让商家尽可能通过智能助手直接得到答案或操作指引，减少转人工工单。
                只允许使用白名单内的分类、回答模式、指标字段和身份字段。
                不要编写 SQL，不要回答用户，必须只输出 JSON 对象。
                """;
    }

    private String buildPrompt(String question, QuestionIntent baseIntent, List<Map<String, Object>> historyRows, String wiki) {
        return """
                用户问题：
                %s

                规则识别结果：
                intentType=%s, category=%s, answerMode=%s, metricColumn=%s, identityColumn=%s, days=%d

                可选 intentType：GREETING, VALID, INVALID
                可选 category：PLATFORM_RULE, TRADE, REFUND, CS_TICKET, COMPENSATION, COUPON, GOODS, MERCHANT_OTHER, IDENTITY, SCM, UNKNOWN
                可选 answerMode：METRIC, DETAIL, RULE, IDENTITY, CHAT, INVALID

                指标字段白名单：
                %s

                身份字段白名单：
                %s

                历史问答参考：
                %s

                wiki 记忆参考：
                %s

                输出 JSON 字段：
                {
                  "intentType": "VALID",
                  "category": "REFUND",
                  "answerMode": "METRIC",
                  "metricColumn": "return_cnt_1d",
                  "identityColumn": "",
                  "days": 7,
                  "reason": "一句话说明判断依据"
                }
                """.formatted(
                question,
                baseIntent.getIntentType(),
                baseIntent.getCategory(),
                baseIntent.getAnswerMode(),
                baseIntent.getMetricColumn(),
                baseIntent.getIdentityColumn(),
                baseIntent.getDays(),
                METRIC_COLUMNS.keySet(),
                IDENTITY_COLUMNS.keySet(),
                summarizeHistory(historyRows),
                truncate(wiki, MAX_WIKI_CHARS)
        );
    }

    private String buildPlanPrompt(String question, QueryPlan fallbackPlan, List<Map<String, Object>> historyRows, String wiki) {
        return """
                用户问题：
                %s

                规则拆分结果：
                %s

                任务要求：
                1. 先理解整句问题，再决定是否需要拆成多个子意图。
                2. 如果只是单一问题，只返回 1 个 intent。
                3. 如果包含并列查询、分别查看、综合经营分析，可拆成多个 intent。
                4. 每个 intent 都必须落在白名单内，不要输出 SQL。

                可选 intentType：GREETING, VALID, INVALID
                可选 category：PLATFORM_RULE, TRADE, REFUND, CS_TICKET, COMPENSATION, COUPON, GOODS, MERCHANT_OTHER, IDENTITY, SCM, UNKNOWN
                可选 answerMode：METRIC, DETAIL, RULE, IDENTITY, CHAT, INVALID

                指标字段白名单：
                %s

                身份字段白名单：
                %s

                历史问答参考：
                %s

                wiki 记忆参考：
                %s

                输出 JSON 字段：
                {
                  "intents": [
                    {
                      "intentType": "VALID",
                      "category": "REFUND",
                      "answerMode": "METRIC",
                      "metricColumn": "refund_amt_1d",
                      "identityColumn": "",
                      "days": 7,
                      "question": "最近7天退款金额"
                    },
                    {
                      "intentType": "VALID",
                      "category": "CS_TICKET",
                      "answerMode": "METRIC",
                      "metricColumn": "cs_ticket_cnt_1d",
                      "identityColumn": "",
                      "days": 7,
                      "question": "最近7天咨询工单量"
                    }
                  ]
                }
                """.formatted(
                question,
                summarizePlan(fallbackPlan),
                METRIC_COLUMNS.keySet(),
                IDENTITY_COLUMNS.keySet(),
                summarizeHistory(historyRows),
                truncate(wiki, MAX_WIKI_CHARS)
        );
    }

    private QuestionIntent applyJsonIntent(String question, QuestionIntent baseIntent, String json) throws Exception {
        JsonNode root = objectMapper.readTree(json);
        QuestionIntent intent = copy(baseIntent);
        intent.setQuestion(question);

        IntentType intentType = parseIntentType(text(root, "intentType"), baseIntent.getIntentType());
        QuestionCategory category = parseCategory(text(root, "category"), baseIntent.getCategory());
        AnswerMode answerMode = parseAnswerMode(text(root, "answerMode"), baseIntent.getAnswerMode());
        int days = root.path("days").asInt(baseIntent.getDays());

        if (intentType == IntentType.GREETING) {
            intent.setIntentType(IntentType.GREETING);
            intent.setCategory(QuestionCategory.UNKNOWN);
            intent.setAnswerMode(AnswerMode.CHAT);
            return intent;
        }
        if (intentType == IntentType.INVALID && category == QuestionCategory.UNKNOWN) {
            intent.setIntentType(IntentType.INVALID);
            intent.setCategory(QuestionCategory.UNKNOWN);
            intent.setAnswerMode(AnswerMode.INVALID);
            return intent;
        }

        intent.setIntentType(category == QuestionCategory.UNKNOWN ? intentType : IntentType.VALID);
        intent.setCategory(category);
        intent.setDays(days);
        intent.setAnswerMode(normalizeMode(category, answerMode));
        applyMetric(root, intent);
        applyIdentity(root, intent);
        return intent;
    }

    private void applyMetric(JsonNode root, QuestionIntent intent) {
        String metricColumn = text(root, "metricColumn");
        MetricDefinition metric = METRIC_COLUMNS.get(metricColumn);
        if (metric == null && intent.getAnswerMode() == AnswerMode.METRIC) {
            metric = DEFAULT_METRICS.get(intent.getCategory());
        }
        if (metric != null) {
            intent.setMetricColumn(metric.getColumn());
            intent.setMetricName(metric.getDisplayName());
            intent.setMetricUnit(metric.getUnit());
        }
    }

    private void applyIdentity(JsonNode root, QuestionIntent intent) {
        String identityColumn = text(root, "identityColumn");
        if (IDENTITY_COLUMNS.containsKey(identityColumn)) {
            intent.setIdentityColumn(identityColumn);
            intent.setIdentityName(IDENTITY_COLUMNS.get(identityColumn));
        }
    }

    private QueryPlan applyPlanJson(String question, String json) throws Exception {
        JsonNode root = objectMapper.readTree(json);
        JsonNode intentsNode = root.path("intents");
        QueryPlan plan = new QueryPlan();
        Set<String> signatures = new LinkedHashSet<>();
        if (!intentsNode.isArray()) {
            return plan;
        }
        for (JsonNode item : intentsNode) {
            QuestionIntent intent = applyJsonIntent(itemQuestion(question, item), new QuestionIntent(), item.toString());
            String signature = planSignature(intent);
            if (signatures.add(signature)) {
                plan.getIntents().add(intent);
            }
        }
        return plan;
    }

    private QueryPlan sanitizeRefinedPlan(QueryPlan refined, QueryPlan fallback) {
        if (refined == null || refined.getIntents().isEmpty()) {
            return fallback;
        }

        long fallbackValidCount = fallback.getIntents().stream()
                .filter(intent -> intent.getIntentType() == IntentType.VALID)
                .count();
        long refinedValidCount = refined.getIntents().stream()
                .filter(intent -> intent.getIntentType() == IntentType.VALID)
                .count();

        if (fallbackValidCount > 0 && refinedValidCount < fallbackValidCount) {
            return fallback;
        }
        if (refinedValidCount == refined.getIntents().size()) {
            return refined;
        }

        QueryPlan validOnly = new QueryPlan();
        for (QuestionIntent intent : refined.getIntents()) {
            if (intent.getIntentType() == IntentType.VALID) {
                validOnly.getIntents().add(intent);
            }
        }
        return validOnly.getIntents().isEmpty() ? fallback : validOnly;
    }

    private AnswerMode normalizeMode(QuestionCategory category, AnswerMode answerMode) {
        if (category == QuestionCategory.PLATFORM_RULE) {
            return AnswerMode.RULE;
        }
        if (category == QuestionCategory.IDENTITY) {
            return AnswerMode.IDENTITY;
        }
        if (answerMode == AnswerMode.DETAIL || answerMode == AnswerMode.METRIC) {
            return answerMode;
        }
        return AnswerMode.METRIC;
    }

    private IntentType parseIntentType(String value, IntentType fallback) {
        if ("GREETING".equalsIgnoreCase(value)) {
            return IntentType.GREETING;
        }
        if ("VALID".equalsIgnoreCase(value)) {
            return IntentType.VALID;
        }
        if ("INVALID".equalsIgnoreCase(value)) {
            return IntentType.INVALID;
        }
        return fallback;
    }

    private QuestionCategory parseCategory(String value, QuestionCategory fallback) {
        if (!StringUtils.hasText(value)) {
            return fallback;
        }
        for (QuestionCategory category : QuestionCategory.values()) {
            if (category.name().equalsIgnoreCase(value) || category.getDisplayName().equals(value)) {
                return category;
            }
        }
        if (value.contains("规则")) return QuestionCategory.PLATFORM_RULE;
        if (value.contains("交易") || value.contains("订单")) return QuestionCategory.TRADE;
        if (value.contains("退货") || value.contains("退款")) return QuestionCategory.REFUND;
        if (value.contains("客服") || value.contains("工单")) return QuestionCategory.CS_TICKET;
        if (value.contains("赔付") || value.contains("理赔")) return QuestionCategory.COMPENSATION;
        if (value.contains("优惠") || value.contains("券")) return QuestionCategory.COUPON;
        if (value.contains("商品")) return QuestionCategory.GOODS;
        if (value.contains("商家")) return QuestionCategory.MERCHANT_OTHER;
        if (value.contains("身份")) return QuestionCategory.IDENTITY;
        if (value.contains("供应链") || value.contains("质检") || value.contains("入库")) return QuestionCategory.SCM;
        return fallback;
    }

    private AnswerMode parseAnswerMode(String value, AnswerMode fallback) {
        if (!StringUtils.hasText(value)) {
            return fallback;
        }
        for (AnswerMode mode : AnswerMode.values()) {
            if (mode.name().equalsIgnoreCase(value)) {
                return mode;
            }
        }
        if (value.contains("明细") || value.contains("详情")) return AnswerMode.DETAIL;
        if (value.contains("指标") || value.contains("趋势")) return AnswerMode.METRIC;
        if (value.contains("规则")) return AnswerMode.RULE;
        if (value.contains("身份")) return AnswerMode.IDENTITY;
        return fallback;
    }

    private String text(JsonNode root, String field) {
        JsonNode node = root.path(field);
        if (node.isMissingNode() || node.isNull()) {
            return "";
        }
        return node.asText("").trim();
    }

    private String extractJson(String response) {
        String text = response.trim();
        int start = text.indexOf('{');
        int end = text.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return text.substring(start, end + 1);
        }
        return text;
    }

    private QuestionIntent copy(QuestionIntent source) {
        QuestionIntent target = new QuestionIntent();
        target.setIntentType(source.getIntentType());
        target.setCategory(source.getCategory());
        target.setAnswerMode(source.getAnswerMode());
        target.setQuestion(source.getQuestion());
        target.setMetricColumn(source.getMetricColumn());
        target.setMetricName(source.getMetricName());
        target.setMetricUnit(source.getMetricUnit());
        target.setIdentityColumn(source.getIdentityColumn());
        target.setIdentityName(source.getIdentityName());
        target.setDays(source.getDays());
        target.setLlmAnalysisRequested(source.isLlmAnalysisRequested());
        target.setLlmAnalyzed(source.isLlmAnalyzed());
        target.setAnalysisSource(source.getAnalysisSource());
        target.setAnalysisNote(source.getAnalysisNote());
        return target;
    }

    private String summarizeHistory(List<Map<String, Object>> historyRows) {
        if (historyRows == null || historyRows.isEmpty()) {
            return "无";
        }
        StringBuilder builder = new StringBuilder();
        historyRows.stream().limit(MAX_HISTORY_ROWS).forEach(row -> {
            Object question = row.getOrDefault("question", row.getOrDefault("user_question", ""));
            Object category = row.getOrDefault("question_category_name", row.getOrDefault("category_name", ""));
            if (StringUtils.hasText(String.valueOf(question))) {
                builder.append("- [").append(category).append("] ").append(question).append("\n");
            }
        });
        return builder.isEmpty() ? "无" : builder.toString();
    }

    private String summarizePlan(QueryPlan plan) {
        if (plan == null || plan.getIntents().isEmpty()) {
            return "无";
        }
        StringBuilder builder = new StringBuilder();
        for (QuestionIntent intent : plan.getIntents()) {
            builder.append("- ")
                    .append(intent.getCategory())
                    .append(" / ")
                    .append(intent.getAnswerMode())
                    .append(" / metric=")
                    .append(intent.getMetricColumn())
                    .append(" / identity=")
                    .append(intent.getIdentityColumn())
                    .append(" / days=")
                    .append(intent.getDays())
                    .append("\n");
        }
        return builder.toString();
    }

    private QueryPlan copyPlan(QueryPlan source) {
        QueryPlan target = new QueryPlan();
        if (source == null) {
            return target;
        }
        for (QuestionIntent intent : source.getIntents()) {
            target.getIntents().add(copy(intent));
        }
        return target;
    }

    private void markPlanAsRequested(QueryPlan plan) {
        for (QuestionIntent intent : plan.getIntents()) {
            intent.setLlmAnalysisRequested(true);
            if (!StringUtils.hasText(intent.getAnalysisNote())) {
                intent.setAnalysisNote("复杂问题需要大模型拆解，已回退到规则查询计划");
            }
        }
    }

    private void markPlanAsAnalyzed(QueryPlan plan) {
        for (QuestionIntent intent : plan.getIntents()) {
            intent.setLlmAnalysisRequested(true);
            intent.setLlmAnalyzed(true);
            intent.setAnalysisSource("LLM");
            intent.setAnalysisNote("复杂问题已由大模型理解后拆解为查询计划");
        }
    }

    private String itemQuestion(String fallbackQuestion, JsonNode item) {
        String itemQuestion = text(item, "question");
        return StringUtils.hasText(itemQuestion) ? itemQuestion : fallbackQuestion;
    }

    private String planSignature(QuestionIntent intent) {
        return "%s|%s|%s|%s|%d".formatted(
                intent.getIntentType(),
                intent.getCategory(),
                intent.getAnswerMode(),
                intent.getMetricColumn() + ":" + intent.getIdentityColumn(),
                intent.getDays()
        );
    }

    private String truncate(String text, int maxLength) {
        if (text == null || text.length() <= maxLength) {
            return text == null ? "" : text;
        }
        return text.substring(0, maxLength);
    }
}
