package com.yshopping.merchantai.service;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.IntentType;
import com.yshopping.merchantai.model.MerchantInfo;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.model.QueryBundle;
import com.yshopping.merchantai.model.QuestionIntent;
import com.yshopping.merchantai.model.RuleTopic;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * 回答组织服务。
 *
 * <p>负责把意图、Doris 结果、商家信息和 wiki 记忆整理成给商家的自然语言话术，
 * 把“自动理解”后的结构化结果真正转成“自动回复”和“操作指引”；LLM 不可用时
 * 使用本地降级回答。</p>
 */
public class AnswerComposeService {
    private final LlmClient llmClient;
    private final ObjectMapper objectMapper;

    public AnswerComposeService(LlmClient llmClient, ObjectMapper objectMapper) {
        this.llmClient = llmClient;
        this.objectMapper = objectMapper;
    }

    public String greeting() {
        return "您好，我是 yshopping 商家 AI 助手，可以帮您处理订单物流、商品上架、退款售后、费率规则、平台操作指引、客服工单和经营数据问题。";
    }

    public String invalid() {
        return "我暂时还没准确理解您的问题。您可以直接描述您要处理的场景，例如物流单号关联订单、商品上架、退款处理、费率规则、平台操作或工单进度；如果仍无法处理，再转人工工单。";
    }

    public String appendBusinessAdvice(String answer, List<QuestionIntent> intents, QueryBundle queryBundle) {
        if (!StringUtils.hasText(answer) || answer.contains("经营建议")) {
            return answer;
        }
        List<QuestionIntent> validIntents = intents == null ? List.of() : intents.stream()
                .filter(intent -> intent != null && intent.getIntentType() == IntentType.VALID)
                .toList();
        if (validIntents.isEmpty()) {
            return answer;
        }
        List<String> advice = businessAdvice(validIntents, queryBundle);
        if (advice.isEmpty()) {
            return answer;
        }
        StringBuilder builder = new StringBuilder(answer.trim());
        builder.append("\n\n经营建议");
        for (int i = 0; i < Math.min(2, advice.size()); i++) {
            builder.append("\n").append(i + 1).append(". ").append(advice.get(i));
        }
        return builder.toString();
    }

    public String compose(String question, MerchantInfo merchant, QuestionIntent intent, QueryBundle queryBundle, String wiki) {
        if (intent.getAnswerMode() == AnswerMode.RULE) {
            return composeRuleAnswer(question, wiki);
        }
        if (intent.getAnswerMode() == AnswerMode.IDENTITY) {
            return composeIdentityAnswer(question, intent, queryBundle);
        }
        if (intent.getAnswerMode() == AnswerMode.DETAIL) {
            return composeDetailAnswer(merchant, intent, queryBundle);
        }
        String fallback = fallbackAnswer(intent, queryBundle);
        String prompt = """
                用户问题：%s
                商家：%s（merchant_id=%s）
                问题分类：%s
                回答模式：%s
                指标：%s
                时间范围：最近 %d 天
                Doris 表：%s
                Doris 结果：
                %s
                指标汇总：
                %s

                可用 wiki：
                %s
                """.formatted(
                question,
                merchant.getMerchantName(),
                merchant.getMerchantId(),
                intent.getCategory().getDisplayName(),
                intent.getAnswerMode(),
                intent.getMetricName(),
                intent.getDays(),
                queryBundle.getTables(),
                toJson(queryBundle.getRows()),
                metricSummary(intent, queryBundle),
                wiki
        );
        return llmClient.chat(systemPrompt(), prompt, fallback);
    }

    private List<String> businessAdvice(List<QuestionIntent> intents, QueryBundle queryBundle) {
        Set<String> advice = new LinkedHashSet<>();
        if (hasMode(intents, AnswerMode.METRIC) && queryBundle != null && !queryBundle.getRows().isEmpty()) {
            advice.add("优先关注数值偏高或波动明显的日期，再继续查看对应明细，定位具体订单、工单或商品记录。");
        }
        if (hasMode(intents, AnswerMode.DETAIL)) {
            advice.add("建议先处理状态异常、时间较近或备注明确的记录，避免问题继续积压影响履约和售后体验。");
        }
        if (hasMode(intents, AnswerMode.RULE)) {
            advice.add("建议按规则先补齐资料、类目、图文描述和操作凭证，再提交审核或发起处理，减少反复驳回。");
        }
        if (hasMode(intents, AnswerMode.IDENTITY)) {
            advice.add("建议定期核对商家资料、保证金、地址和结算信息，避免因资料不一致影响发货、退款或结算。");
        }
        for (QuestionIntent intent : intents) {
            switch (intent.getCategory()) {
                case TRADE -> advice.add("若订单或 GMV 出现异常，建议结合订单明细查看支付状态、发货状态和超时记录。");
                case REFUND -> advice.add("若退货退款量偏高，建议优先查看退款原因和卖家责任记录，针对高频原因优化商品描述或售后口径。");
                case CS_TICKET -> advice.add("若咨询工单量偏高，建议重点排查物流停滞、催单、商品说明不清和售后处理时效。");
                case GOODS -> advice.add("若商品审核或上架存在异常，建议优先核对类目、品牌资质、主图详情和审核备注后再重新提交。");
                case COMPENSATION -> advice.add("若赔付金额或赔付单量偏高，建议结合赔付原因区分物流、售后和商品质量责任。");
                case COUPON -> advice.add("若优惠金额占比偏高，建议结合交易成功优惠和支付优惠数据，评估活动成本与转化效果。");
                case MERCHANT_OTHER -> advice.add("若涉及保证金、申诉或处罚，建议先核对资金流水和平台备注，再决定是否补缴或发起申诉。");
                case SCM -> advice.add("若供应链履约异常，建议优先查看入库、质检、鉴定和出库节点，定位卡点环节。");
                case PLATFORM_RULE -> advice.add("遇到规则类问题时，建议先对照平台规则完成自查，再结合业务明细确认是否需要补充资料。");
                default -> {
                }
            }
            if (advice.size() >= 2) {
                break;
            }
        }
        if (advice.size() < 2) {
            advice.add("如果需要进一步定位，可以继续追问对应的明细、异常原因或最近 7 天趋势。");
        }
        return advice.stream().limit(2).toList();
    }

    private boolean hasMode(List<QuestionIntent> intents, AnswerMode mode) {
        return intents.stream().anyMatch(intent -> intent.getAnswerMode() == mode);
    }

    private String composeDetailAnswer(MerchantInfo merchant, QuestionIntent intent, QueryBundle queryBundle) {
        if (queryBundle.getRows().isEmpty()) {
            return fallbackAnswer(intent, queryBundle);
        }
        if (intent.getCategory() == QuestionCategory.GOODS) {
            return composeGoodsDetailAnswer(merchant, intent, queryBundle);
        }
        return fallbackAnswer(intent, queryBundle);
    }

    private String composeIdentityAnswer(String question, QuestionIntent intent, QueryBundle queryBundle) {
        // 身份信息只回答商家要问的字段，不把 dim_merchant_df 伪装成“明细列表”展示。
        String fallback = identityFallback(intent, queryBundle);
        String prompt = """
                用户正在咨询 yshopping 商家身份资料。
                用户问题：%s
                识别字段：%s（%s）
                格式化字段回答：%s
                Doris 查询结果：
                %s

                请优先使用“格式化字段回答”，直接回答该字段的值，话术格式类似“您的公司名称为 XXX。”；
                如果未查到数据，提示商家确认商家 ID 或稍后重试。不要输出表名、SQL 或 JSON。
                """.formatted(
                question,
                intent.getIdentityName(),
                intent.getIdentityColumn(),
                fallback,
                toJson(queryBundle.getRows())
        );
        return llmClient.chat(systemPrompt(), prompt, fallback);
    }

    public List<String> suggestions(String categoryName, List<Map<String, Object>> historyQuestions) {
        Set<String> merged = new LinkedHashSet<>();
        if (historyQuestions != null && !historyQuestions.isEmpty()) {
            historyQuestions.stream()
                    .map(row -> String.valueOf(row.getOrDefault("question", "")))
                    .filter(text -> !text.isBlank())
                    .forEach(merged::add);
        }
        merged.addAll(defaultSuggestions(categoryName));
        return new ArrayList<>(merged);
    }

    public List<String> ruleSuggestions(RuleTopic topic) {
        return switch (topic) {
            case GOODS_CATEGORY -> List.of("商品类目怎么选？", "放错商品类目会有什么影响？", "商品类目和资质有什么关系？");
            case GOODS_QUALIFICATION -> List.of("上架商品需要哪些资质？", "品牌商品需要授权资质吗？", "资质不全会影响商品审核吗？");
            case GOODS_CONTENT -> List.of("商品主图有哪些要求？", "商品详情页描述怎么写？", "图文不符会导致审核被拒吗？");
            case GOODS_REJECT -> List.of("商品审核被拒怎么办？", "怎么查看商品审核拒绝原因？", "修改后怎么重新提交审核？");
            case GOODS_LISTING -> List.of("商品上架要求是什么？", "商品类目规范是什么？", "商品审核被拒怎么办？");
            case DELIVERY -> List.of("发货规则有哪些？", "超时发货怎么处理？", "物流单号怎么关联订单？");
            case REFUND -> List.of("退款处理规则是什么？", "直接退款是什么意思？", "卖家责任退货怎么判断？");
            case CS_TICKET -> List.of("工单进度怎么查？", "工单二次开启是什么意思？", "催单太多怎么办？");
            case COMPENSATION -> List.of("赔付规则是什么？", "赔付原因怎么查？", "赔付可以申诉吗？");
            case COUPON -> List.of("优惠券规则有哪些？", "优惠金额怎么算？", "优惠券明细怎么查？");
            case DEPOSIT -> List.of("保证金规则是什么？", "冻结保证金是什么意思？", "保证金充值记录怎么查？");
            case GENERAL -> List.of("商品上架要求是什么？", "退款处理规则是什么？", "工单进度怎么查？");
        };
    }

    private List<String> defaultSuggestions(String categoryName) {
        return switch (categoryName) {
            case "平台商家规则" -> List.of("商品上架要求是什么？", "退款处理规则是什么？", "工单进度怎么查？", "保证金规则是什么？");
            case "电商交易" -> List.of("昨天总 GMV 是多少？", "最近7天交易成功订单量趋势", "帮我看一下订单明细", "昨天发货超时订单量是多少？");
            case "电商退货" -> List.of("最近7天退货量趋势", "昨天退款金额是多少？", "查看退货明细", "直接退款量是多少？");
            case "电商客服工单" -> List.of("最近7天咨询工单量", "工单二次开启量是多少？", "查看工单明细", "催单工单量是多少？");
            case "电商理赔/赔付" -> List.of("昨天卖家赔付金额是多少？", "查看赔付单明细", "最近7天赔付订单数趋势", "赔付订单数是多少？");
            case "电商优惠券" -> List.of("交易成功优惠金额是多少？", "查看优惠券明细", "最近7天支付成功优惠单量", "优惠金额占比是多少？");
            case "商品管理" -> List.of("昨天商品审核拒绝量是多少？", "查看商品上架明细", "最近7天上架商品量", "商品审核通过量是多少？");
            case "商家其他信息" -> List.of("保证金余额是多少", "最近7天申诉次数", "查看保证金充值记录", "最近7天处罚次数");
            case "身份信息" -> List.of("保证金余额是多少", "查看冻结保证金", "查看发货地址", "查看银行卡信息");
            default -> List.of("保证金余额是多少", "我要货品上架，具体规则有吗？", "最近7天咨询工单量", "查看商品上架明细");
        };
    }

    private String identityFallback(QuestionIntent intent, QueryBundle queryBundle) {
        if (queryBundle.getRows().isEmpty()) {
            String fieldName = StringUtils.hasText(intent.getIdentityName()) ? intent.getIdentityName() : "商家身份信息";
            return "当前未查到您的%s，请确认商家 ID 或稍后重试。".formatted(fieldName);
        }
        Map<String, Object> row = queryBundle.getRows().get(0);
        if (StringUtils.hasText(intent.getIdentityColumn())) {
            Object value = rowValue(row, intent.getIdentityColumn());
            String fieldName = StringUtils.hasText(intent.getIdentityName()) ? intent.getIdentityName() : "信息";
            if (value == null || String.valueOf(value).isBlank()) {
                return "当前未查到您的%s，请确认资料是否已维护。".formatted(fieldName);
            }
            if ("is_invoice".equals(intent.getIdentityColumn())) {
                return enabled(value) ? "您当前支持开具发票。" : "您当前不支持开具发票。";
            }
            if ("is_unconditional_refund".equals(intent.getIdentityColumn())) {
                return enabled(value) ? "您当前支持七天无理由退货。" : "您当前不支持七天无理由退货。";
            }
            if ("poundage_discount".equals(intent.getIdentityColumn())) {
                return "您的当前费率折扣为%s。".formatted(value);
            }
            if (isAmountColumn(intent.getIdentityColumn())) {
                return "您的%s为%s元。".formatted(fieldName, value);
            }
            return "您的%s为%s。".formatted(fieldName, value);
        }
        return """
                我查到了您的商家身份信息：
                - 公司名称：%s
                - 商户类型：%s
                - 结算类型：%s
                - 联系人：%s
                """.formatted(
                rowValueOrDefault(row, "company_name", "-"),
                rowValueOrDefault(row, "merchant_type_name", "-"),
                rowValueOrDefault(row, "balance_type_name", "-"),
                rowValueOrDefault(row, "contact_name", "-")
        ).trim();
    }

    private Object rowValueOrDefault(Map<String, Object> row, String column, Object defaultValue) {
        Object value = rowValue(row, column);
        return value == null || String.valueOf(value).isBlank() ? defaultValue : value;
    }

    private Object rowValue(Map<String, Object> row, String column) {
        if (row.containsKey(column)) {
            return row.get(column);
        }
        for (Map.Entry<String, Object> entry : row.entrySet()) {
            if (entry.getKey() != null && entry.getKey().equalsIgnoreCase(column)) {
                return entry.getValue();
            }
        }
        return null;
    }

    private boolean enabled(Object value) {
        if (value instanceof Number number) {
            return number.longValue() == 1L;
        }
        String text = String.valueOf(value).trim();
        return "1".equals(text) || "true".equalsIgnoreCase(text) || "是".equals(text);
    }

    private boolean isAmountColumn(String column) {
        return "deposit_amt".equals(column)
                || "deposit_freeze".equals(column)
                || "init_deposit_amt".equals(column)
                || "min_poundage".equals(column)
                || "max_poundage".equals(column);
    }

    private String composeRuleAnswer(String question, String wiki) {
        RuleTopic topic = ruleTopicFromQuestion(question);
        String topicAnswer = ruleTopicAnswer(topic);
        String prompt = """
                用户在询问平台规则或商家培训信息。
                用户问题：%s
                识别到的规则主题：%s
                推荐回答要点：
                %s

                规则文档：
                %s

                请用 yshopping 商家助手口吻回答。优先围绕识别到的规则主题作答，先直接回答规则是否有，再给出可操作步骤和注意事项。
                """.formatted(question, topic.getDisplayName(), topicAnswer, wiki);
        String fallback = topicAnswer;
        return llmClient.chat(systemPrompt(), prompt, fallback);
    }

    private RuleTopic ruleTopicFromQuestion(String question) {
        String text = question == null ? "" : question;
        if (containsAny(text, "类目", "商品类目", "类目规范", "类目要求", "类目怎么选", "放错类目")) {
            return RuleTopic.GOODS_CATEGORY;
        }
        if (containsAny(text, "资质", "资质要求", "需要哪些资质", "品牌资质", "授权资质", "经营资质")) {
            return RuleTopic.GOODS_QUALIFICATION;
        }
        if (containsAny(text, "图文要求", "主图要求", "详情页要求", "图片要求", "描述要求", "图文描述")) {
            return RuleTopic.GOODS_CONTENT;
        }
        if (containsAny(text, "审核被拒", "拒绝原因", "审核拒绝", "被驳回", "重新提交")) {
            return RuleTopic.GOODS_REJECT;
        }
        if (containsAny(text, "商品", "上架", "货品", "spu", "SPU", "sku", "SKU", "审核")) {
            return RuleTopic.GOODS_LISTING;
        }
        if (containsAny(text, "发货", "签收", "物流", "超时", "订单")) {
            return RuleTopic.DELIVERY;
        }
        if (containsAny(text, "退货", "退款", "退换", "售后")) {
            return RuleTopic.REFUND;
        }
        if (containsAny(text, "客服", "工单", "咨询", "催单")) {
            return RuleTopic.CS_TICKET;
        }
        if (containsAny(text, "理赔", "赔付", "赔款", "补偿")) {
            return RuleTopic.COMPENSATION;
        }
        if (containsAny(text, "优惠券", "优惠", "券", "补贴", "折扣")) {
            return RuleTopic.COUPON;
        }
        if (containsAny(text, "保证金", "缴纳", "充值", "冻结")) {
            return RuleTopic.DEPOSIT;
        }
        return RuleTopic.GENERAL;
    }

    private String ruleTopicAnswer(RuleTopic topic) {
        return switch (topic) {
            case GOODS_LISTING -> """
                    有的。货品/商品上架通常要先完成商品信息补全，再进入商品审核和风险审核。

                    建议您上架前重点确认这些信息：
                    - 商品类目、品牌资质、货号、标题和图文描述填写完整且一致。
                    - 商品图片、视频、描述不要和实际货品不符，避免审核被拒。
                    - 如果审核被拒，优先查看拒绝原因、类目、品牌资质、图文描述、货号和风控备注，再按原因修改后重新提交。

                    如果您想定位具体问题，可以继续问“查看商品上架明细”或“昨天商品审核拒绝原因”。
                    """.trim();
            case GOODS_CATEGORY -> """
                    商品类目规范主要是：上架时选择的类目必须和实际售卖商品一致，不能错放、混放，也不能通过放到无关类目来规避审核。

                    您可以重点确认这些点：
                    - 商品类目要和商品实际属性、用途、材质或规格一致。
                    - 不同类目通常会对应不同的品牌资质、商品属性和发布要求。
                    - 如果类目选错，容易导致审核被拒、属性填写不匹配，严重时还会影响后续经营。

                    如果您不确定当前商品放在哪个类目，建议先核对商品属性和平台要求，再补齐对应资质后提交。
                    """.trim();
            case GOODS_QUALIFICATION -> """
                    上架商品是否需要资质，通常取决于商品类目、品牌属性和平台准入要求。

                    一般建议您重点确认：
                    - 当前类目是否要求品牌授权、经营许可或行业资质。
                    - 商品类目、品牌信息、商品属性和提交资料是否一致。
                    - 如果属于品牌商品、特殊经营商品或准入类目，通常要先补齐对应资质再提交审核。

                    如果您是想定位具体问题，也可以继续问“商品类目规范是什么”或“商品审核被拒怎么办”。
                    """.trim();
            case GOODS_CONTENT -> """
                    商品图文规范主要是要求主图、详情页、视频、标题和商品描述与实际售卖商品保持一致，不能夸大宣传，也不能图文不符。

                    上架前建议重点检查：
                    - 商品标题、主图、详情描述、规格参数是否一致。
                    - 图片、视频和文案是否准确展示商品卖点、型号和属性。
                    - 是否存在虚假宣传、夸张承诺、误导性表述或无关素材。

                    如果审核被拒，建议优先结合审核备注查看是否与图文描述或商品属性有关。
                    """.trim();
            case GOODS_REJECT -> """
                    如果商品审核被拒，建议先不要重复提交，先定位拒绝原因再修改。

                    一般优先排查这些项：
                    - 商品类目是否选对。
                    - 品牌资质、授权资料或行业资质是否齐全。
                    - 商品标题、主图、详情页、视频和规格参数是否与实际商品一致。
                    - 是否有风控备注、限制类目或不符合平台要求的内容。

                    确认原因后，按审核备注修改资料再重新提交，通常比反复提交更有效。
                    """.trim();
            case DELIVERY -> "有的。交易发货规则建议重点关注订单发货时效、物流信息准确性和签收超时情况；若出现发货或签收异常，可以结合订单明细和供应链明细定位原因。";
            case REFUND -> "有的。退货退款规则建议先区分直接退款、退货成功、卖家责任退货和退款金额口径；处理时优先查看退款原因、订单状态和售后记录。";
            case CS_TICKET -> "有的。客服工单规则建议重点关注响应时效、催单、二次开启和评价分；工单异常时优先检查物流进度、售后口径和商品说明。";
            case COMPENSATION -> "有的。理赔赔付规则需要区分赔付订单数、赔付金额、赔付方式和赔付原因；建议先查看赔付明细再判断责任归因。";
            case COUPON -> "有的。优惠券规则建议关注券模板、适用门槛、发放状态、优惠金额和活动有效期；查看效果时可以结合支付成功优惠和交易成功优惠口径。";
            case DEPOSIT -> "有的。保证金规则建议关注保证金余额、冻结保证金、入驻初始保证金和充值记录；如果要核对流水，可以查询保证金充值明细。";
            case GENERAL -> "我可以帮您查询 yshopping 平台商家规则。当前建议您先确认问题所属环节，例如商品上架、交易发货、退货退款、客服工单、理赔赔付、优惠券或保证金，再按规则准备资料和操作。";
        };
    }

    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (keyword != null && !keyword.isBlank() && text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private String fallbackAnswer(QuestionIntent intent, QueryBundle queryBundle) {
        if (queryBundle.getRows().isEmpty()) {
            return "我已识别到您咨询的是「%s」，但当前没有查到对应数据。建议您确认日期范围或稍后重试。".formatted(intent.getCategory().getDisplayName());
        }
        if (intent.getAnswerMode() == AnswerMode.METRIC) {
            StringBuilder builder = new StringBuilder();
            builder.append("我查到了「").append(intent.getMetricName()).append("」最近 ")
                    .append(intent.getDays()).append(" 天的数据：");
            double total = 0D;
            for (Map<String, Object> row : queryBundle.getRows()) {
                double value = number(row.get("value"));
                total += value;
                builder.append("\n- ").append(row.get("pt")).append("：").append(format(value)).append(intent.getMetricUnit());
            }
            if (intent.getDays() > 1) {
                builder.append("\n合计：").append(format(total)).append(intent.getMetricUnit()).append("。");
            }
            builder.append("\n建议您结合明细数据继续查看异常日期的订单、工单或商品记录。");
            return builder.toString();
        }
        return "我查到了相关明细，共 " + queryBundle.getRows().size() + " 条，本次展示如下表。建议优先关注最近日期、关键状态和需要跟进的记录。";
    }

    private String composeGoodsDetailAnswer(MerchantInfo merchant, QuestionIntent intent, QueryBundle queryBundle) {
        List<Map<String, Object>> rows = queryBundle.getRows();
        int total = rows.size();
        long auditPassCount = rows.stream().filter(row -> number(row.get("is_audit_pass")) >= 1D).count();
        long auditRejectCount = total - auditPassCount;
        long remarkCount = rows.stream()
                .map(row -> String.valueOf(row.getOrDefault("audit_remark", "")).trim())
                .filter(text -> !text.isBlank() && !"-".equals(text))
                .count();
        Map<String, Long> statusDistribution = topCounts(rows, "spu_status_name", 5);
        Map<String, Long> operateDistribution = topCounts(rows, "audit_operate_type_name", 5);
        String period = dateRange(rows);
        String merchantName = StringUtils.hasText(merchant.getMerchantName()) ? merchant.getMerchantName() : "当前商家";

        StringBuilder builder = new StringBuilder();
        builder.append(merchantName)
                .append(" 最近 ")
                .append(intent.getDays())
                .append(" 天（")
                .append(period)
                .append("）的商品申请 / SPU 明细如下：");

        builder.append("\n\n一、总体情况");
        builder.append("\n- 商品申请量：").append(total).append(" 条");
        builder.append("\n- 审核通过 / 通过标记：").append(auditPassCount).append(" 条");
        builder.append("\n- 未通过 / 待处理：").append(auditRejectCount).append(" 条");
        builder.append("\n- 带审核备注记录：").append(remarkCount).append(" 条");

        builder.append("\n\n二、SPU 状态分布");
        appendCountLines(builder, statusDistribution);

        builder.append("\n\n三、审核操作分布");
        appendCountLines(builder, operateDistribution);

        builder.append("\n\n四、明细说明");
        builder.append("\n- 下方表格已按申请时间倒序展示，您可以继续滚动查看完整明细。");
        builder.append("\n- 建议优先关注审核未通过、状态异常或审核备注需要跟进的记录。");
        return builder.toString();
    }

    private Map<String, Long> topCounts(List<Map<String, Object>> rows, String column, int limit) {
        Map<String, Long> counts = new LinkedHashMap<>();
        for (Map<String, Object> row : rows) {
            String key = String.valueOf(row.getOrDefault(column, "")).trim();
            if (key.isBlank()) {
                key = "未标记";
            }
            counts.put(key, counts.getOrDefault(key, 0L) + 1L);
        }
        return counts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue(Comparator.reverseOrder())
                        .thenComparing(Map.Entry.comparingByKey()))
                .limit(limit)
                .collect(LinkedHashMap::new,
                        (map, entry) -> map.put(entry.getKey(), entry.getValue()),
                        LinkedHashMap::putAll);
    }

    private void appendCountLines(StringBuilder builder, Map<String, Long> counts) {
        if (counts.isEmpty()) {
            builder.append("\n- 暂无可用分布数据");
            return;
        }
        counts.forEach((key, value) -> builder.append("\n- ").append(key).append("：").append(value).append(" 条"));
    }

    private String dateRange(List<Map<String, Object>> rows) {
        String min = "";
        String max = "";
        for (Map<String, Object> row : rows) {
            String pt = String.valueOf(row.getOrDefault("pt", "")).trim();
            if (pt.isBlank()) {
                continue;
            }
            if (min.isBlank() || pt.compareTo(min) < 0) {
                min = pt;
            }
            if (max.isBlank() || pt.compareTo(max) > 0) {
                max = pt;
            }
        }
        if (min.isBlank() || max.isBlank()) {
            return "-";
        }
        return min.equals(max) ? min : min + " 至 " + max;
    }

    private String metricSummary(QuestionIntent intent, QueryBundle queryBundle) {
        if (intent.getAnswerMode() != AnswerMode.METRIC || queryBundle.getRows().isEmpty()) {
            return "无";
        }
        double total = 0D;
        StringBuilder daily = new StringBuilder();
        for (Map<String, Object> row : queryBundle.getRows()) {
            double value = number(row.get("value"));
            total += value;
            daily.append(row.get("pt")).append("=").append(format(value)).append(intent.getMetricUnit()).append("; ");
        }
        return "已从 ads_merchant_profile 按 pt 分区取最近 %d 天每日数据，逐日结果：%s最终合计：%s%s。回答必须给出每日值和合计值。"
                .formatted(intent.getDays(), daily, format(total), intent.getMetricUnit());
    }

    private double number(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value == null) {
            return 0D;
        }
        try {
            return Double.parseDouble(String.valueOf(value));
        } catch (NumberFormatException e) {
            return 0D;
        }
    }

    private String format(double value) {
        if (Math.abs(value - Math.rint(value)) < 0.000001) {
            return String.valueOf((long) value);
        }
        return String.format("%.2f", value);
    }

    private String systemPrompt() {
        return """
                你是 yshopping 商家 AI 助手。
                你的目标是帮助商家更快解决经营问题，尽量减少不必要的人工客服介入。
                要求：
                1. 使用自然、专业、简洁的中文回复。
                2. 所有数据必须基于 Doris 查询结果，不要编造。
                3. 最近 N 天超过 1 天时，必须使用 Doris 返回的每日 value 做 sum，输出每日值和最终合计值。
                4. 明细数据只总结关键记录，不泄露无关隐私字段。
                5. 如果问题可自助处理，优先给出下一步操作指引，不要轻易建议提工单。
                6. 结尾给出 1-2 条可执行经营建议。
                """;
    }

    private String toJson(Object value) {
        try {
            return objectMapper.writerWithDefaultPrettyPrinter().writeValueAsString(value);
        } catch (JsonProcessingException e) {
            return String.valueOf(value);
        }
    }
}
