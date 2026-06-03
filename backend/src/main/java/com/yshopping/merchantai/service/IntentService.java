package com.yshopping.merchantai.service;

import com.yshopping.merchantai.model.AnswerMode;
import com.yshopping.merchantai.model.IntentType;
import com.yshopping.merchantai.model.MetricDefinition;
import com.yshopping.merchantai.model.QueryPlan;
import com.yshopping.merchantai.model.QuestionCategory;
import com.yshopping.merchantai.model.QuestionIntent;
import com.yshopping.merchantai.model.RuleTopic;
import java.util.ArrayList;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.springframework.stereotype.Service;

@Service
/**
 * 轻量意图识别服务。
 *
 * <p>把用户自然语言映射到业务分类、回答模式、指标字段和最近 N 天范围。
 * 当前采用关键词规则，后续可替换为 LLM 分类器或向量检索。</p>
 */
public class IntentService {
    private static final Pattern RECENT_DAYS = Pattern.compile("(最近|近|过去|前)\\s*(\\d{1,3})\\s*[天日]");
    private static final Pattern RECENT_WEEKS = Pattern.compile("(最近|近|过去|前)\\s*(\\d{1,2})\\s*(周|星期|礼拜)");
    private final LlmIntentAnalysisService llmIntentAnalysisService;

    public IntentService(LlmIntentAnalysisService llmIntentAnalysisService) {
        this.llmIntentAnalysisService = llmIntentAnalysisService;
    }

    public QuestionIntent recognize(String rawQuestion) {
        return recognize(rawQuestion, List.of(), "");
    }

    public QuestionIntent recognize(String rawQuestion, List<Map<String, Object>> historyRows, String wiki) {
        String question = rawQuestion == null ? "" : rawQuestion.trim();
        QuestionIntent intent = recognizeByRule(question);
        return maybeAnalyzeWithLlm(question, intent, historyRows, wiki);
    }

    public QueryPlan recognizePlan(String rawQuestion, List<Map<String, Object>> historyRows, String wiki) {
        String question = rawQuestion == null ? "" : rawQuestion.trim();
        QueryPlan fallbackPlan = buildRulePlan(question);
        if (!shouldUseLlmForPlan(question, fallbackPlan)) {
            return normalizePlanTimeRange(question, fallbackPlan);
        }
        return normalizePlanTimeRange(question, llmIntentAnalysisService.refinePlan(question, fallbackPlan, historyRows, wiki));
    }

    private QuestionIntent recognizeByRule(String question) {
        QuestionIntent intent = new QuestionIntent();
        intent.setQuestion(question);
        if (isGreeting(question)) {
            intent.setIntentType(IntentType.GREETING);
            intent.setCategory(QuestionCategory.UNKNOWN);
            intent.setAnswerMode(AnswerMode.CHAT);
            return intent;
        }

        QuestionCategory category = detectCategory(question);
        if (category == QuestionCategory.UNKNOWN) {
            intent.setIntentType(IntentType.INVALID);
            intent.setAnswerMode(AnswerMode.INVALID);
            return intent;
        }

        intent.setIntentType(IntentType.VALID);
        intent.setCategory(category);
        intent.setDays(extractDays(question));
        intent.setAnswerMode(detectMode(question, category));
        if (intent.getAnswerMode() == AnswerMode.DETAIL && !hasExplicitTimeRange(question)) {
            intent.setDays(30);
        }

        if (intent.getAnswerMode() == AnswerMode.RULE) {
            intent.setRuleTopic(resolveRuleTopic(question));
        }

        if (intent.getAnswerMode() == AnswerMode.IDENTITY) {
            resolveIdentityField(question, intent);
        }

        MetricDefinition metric = resolveMetric(question, category);
        if (metric != null) {
            intent.setMetricColumn(metric.getColumn());
            intent.setMetricName(metric.getDisplayName());
            intent.setMetricUnit(metric.getUnit());
        }
        intent.setAnalysisNote("规则意图识别命中");
        return intent;
    }

    private boolean isGreeting(String question) {
        String normalized = question.toLowerCase();
        return normalized.matches("^(你好|您好|hi|hello|hey|在吗|嗨|哈喽|早上好|下午好|晚上好)[!！。,.，\\s]*$");
    }

    private QuestionCategory detectCategory(String question) {
        if (isRuleGuidanceQuestion(question) || containsAny(question,
                "规则", "政策", "培训", "怎么使用", "如何使用", "服务说明", "平台要求",
                "资质要求", "类目要求", "类目规范", "图文要求", "审核要求")) {
            return QuestionCategory.PLATFORM_RULE;
        }
        if (isIdentityQuestion(question)) {
            return QuestionCategory.IDENTITY;
        }
        if (containsAny(question,
                "身份", "联系人", "身份证", "营业执照", "银行卡", "银行账号", "公司名称", "商家手机号",
                "退货地址", "发货地址", "经营地址", "商户类型", "商家类型", "资质类型", "结算类型",
                "发货模式", "账户类型", "开户行", "发票", "开具发票", "七天无理由", "无理由退货",
                "冻结保证金", "初始保证金", "入驻保证金", "手续费上限", "手续费下限")) {
            return QuestionCategory.IDENTITY;
        }
        if (containsAny(question, "交易", "订单", "gmv", "GMV", "支付", "成交", "下单", "客单价", "签收", "发货超时")) {
            return QuestionCategory.TRADE;
        }
        if (containsAny(question, "退货", "退款", "退换", "直接退款", "卖家责任")) {
            return QuestionCategory.REFUND;
        }
        if (containsAny(question, "客服", "工单", "咨询", "催单", "评价分", "二次开启")) {
            return QuestionCategory.CS_TICKET;
        }
        if (containsAny(question, "理赔", "赔付", "赔款", "补偿")) {
            return QuestionCategory.COMPENSATION;
        }
        if (containsAny(question, "优惠券", "优惠", "券", "补贴", "折扣")) {
            return QuestionCategory.COUPON;
        }
        if (containsAny(question, "商品", "上架", "审核拒绝", "审核通过", "货品", "spu", "SKU", "sku")) {
            return QuestionCategory.GOODS;
        }
        if (containsAny(question, "保证金", "申诉", "处罚", "入驻", "商家信息", "费率", "结算")) {
            return QuestionCategory.MERCHANT_OTHER;
        }
        if (containsAny(question, "供应链", "入库", "出库", "质检", "鉴定", "仓库", "履约", "分拣")) {
            return QuestionCategory.SCM;
        }
        return QuestionCategory.UNKNOWN;
    }

    private AnswerMode detectMode(String question, QuestionCategory category) {
        if (category == QuestionCategory.PLATFORM_RULE) {
            return AnswerMode.RULE;
        }
        if (category == QuestionCategory.IDENTITY) {
            return AnswerMode.IDENTITY;
        }
        if (containsAny(question,
                "明细", "细节", "详细", "详情", "列表", "完整数据", "全部数据",
                "哪些", "订单号", "工单号", "退款单", "退货单", "赔付单", "券编号", "单号", "记录")) {
            return AnswerMode.DETAIL;
        }
        return AnswerMode.METRIC;
    }

    private int extractDays(String question) {
        if (containsAny(question, "昨日", "昨天")) {
            return 1;
        }
        if (containsAny(question, "近一周", "最近一周", "过去一周")) {
            return 7;
        }
        if (containsAny(question, "近半个月", "最近半个月", "过去半个月")) {
            return 15;
        }
        if (containsAny(question, "近一个月", "最近一个月", "过去一个月")) {
            return 30;
        }
        Matcher dayMatcher = RECENT_DAYS.matcher(question);
        if (dayMatcher.find()) {
            return Integer.parseInt(dayMatcher.group(2));
        }
        Matcher weekMatcher = RECENT_WEEKS.matcher(question);
        if (weekMatcher.find()) {
            return Integer.parseInt(weekMatcher.group(2)) * 7;
        }
        return 1;
    }

    private boolean hasExplicitTimeRange(String question) {
        if (containsAny(question,
                "昨日", "昨天", "今天", "今日",
                "近一周", "最近一周", "过去一周",
                "近半个月", "最近半个月", "过去半个月",
                "近一个月", "最近一个月", "过去一个月")) {
            return true;
        }
        return RECENT_DAYS.matcher(question).find() || RECENT_WEEKS.matcher(question).find();
    }

    private boolean isIdentityQuestion(String question) {
        if (isCurrentDepositQuestion(question)) {
            return true;
        }
        return containsAny(question, "当前费率", "费率折扣", "手续费折扣", "我的费率")
                || (containsAny(question, "费率", "手续费") && !containsAny(question, "最近", "近", "趋势", "次数", "明细", "记录"));
    }

    private boolean isCurrentDepositQuestion(String question) {
        if (!containsAny(question, "保证金")) {
            return false;
        }
        if (containsAny(question, "缴纳", "充值", "补缴", "次数", "记录", "明细", "最近", "近", "趋势")) {
            return false;
        }
        return containsAny(question, "当前", "查看", "查询", "看下", "看一下", "还有多少", "还剩", "剩余", "余额", "多少", "我的", "现有");
    }

    private void resolveIdentityField(String question, QuestionIntent intent) {
        // 身份信息类问题需要回答具体字段，避免把 dim_merchant_df 当明细表直接展示给商家。
        if (containsAny(question, "公司名称", "公司名", "企业名称")) {
            intent.setIdentityColumn("company_name");
            intent.setIdentityName("公司名称");
        } else if (containsAny(question, "商家手机号", "手机号", "联系电话", "联系方式")) {
            intent.setIdentityColumn("mobile");
            intent.setIdentityName("商家手机号");
        } else if (containsAny(question, "联系人")) {
            intent.setIdentityColumn("contact_name");
            intent.setIdentityName("联系人姓名");
        } else if (containsAny(question, "营业执照")) {
            intent.setIdentityColumn("license_id");
            intent.setIdentityName("营业执照编号");
        } else if (containsAny(question, "发货地址")) {
            intent.setIdentityColumn("send_address");
            intent.setIdentityName("发货地址");
        } else if (containsAny(question, "退货地址")) {
            intent.setIdentityColumn("refnd_address");
            intent.setIdentityName("退货地址");
        } else if (containsAny(question, "经营地址")) {
            intent.setIdentityColumn("business_address");
            intent.setIdentityName("经营地址");
        } else if (containsAny(question, "开户行")) {
            intent.setIdentityColumn("bank_name");
            intent.setIdentityName("开户行");
        } else if (containsAny(question, "手续费上限")) {
            intent.setIdentityColumn("max_poundage");
            intent.setIdentityName("手续费上限");
        } else if (containsAny(question, "手续费下限")) {
            intent.setIdentityColumn("min_poundage");
            intent.setIdentityName("手续费下限");
        } else if (containsAny(question, "当前费率", "费率折扣", "手续费折扣", "我的费率", "费率", "手续费")) {
            intent.setIdentityColumn("poundage_discount");
            intent.setIdentityName("费率折扣");
        } else if (containsAny(question, "冻结保证金")) {
            intent.setIdentityColumn("deposit_freeze");
            intent.setIdentityName("冻结保证金");
        } else if (containsAny(question, "初始保证金", "入驻保证金")) {
            intent.setIdentityColumn("init_deposit_amt");
            intent.setIdentityName("入驻初始保证金");
        } else if (isCurrentDepositQuestion(question)) {
            intent.setIdentityColumn("deposit_amt");
            intent.setIdentityName("保证金余额");
        } else if (containsAny(question, "银行卡", "银行账号")) {
            intent.setIdentityColumn("bank_account");
            intent.setIdentityName("银行账号");
        } else if (containsAny(question, "商户类型", "商家类型")) {
            intent.setIdentityColumn("merchant_type_name");
            intent.setIdentityName("商户类型");
        } else if (containsAny(question, "资质类型")) {
            intent.setIdentityColumn("brand_type_name");
            intent.setIdentityName("资质类型");
        } else if (containsAny(question, "结算类型")) {
            intent.setIdentityColumn("balance_type_name");
            intent.setIdentityName("结算类型");
        } else if (containsAny(question, "发货模式")) {
            intent.setIdentityColumn("ship_model_name");
            intent.setIdentityName("发货模式");
        } else if (containsAny(question, "账户类型")) {
            intent.setIdentityColumn("account_type_name");
            intent.setIdentityName("账户类型");
        } else if (containsAny(question, "发票", "开具发票")) {
            intent.setIdentityColumn("is_invoice");
            intent.setIdentityName("开具发票");
        } else if (containsAny(question, "七天无理由", "无理由退货")) {
            intent.setIdentityColumn("is_unconditional_refund");
            intent.setIdentityName("七天无理由退货");
        }
    }

    private MetricDefinition resolveMetric(String question, QuestionCategory category) {
        Map<String, MetricDefinition> candidates = metrics(category);
        for (Map.Entry<String, MetricDefinition> entry : candidates.entrySet()) {
            if (containsAny(question, entry.getKey().split("\\|"))) {
                return entry.getValue();
            }
        }
        return switch (category) {
            case TRADE -> new MetricDefinition("order_cnt_1d", "总订单量", "单");
            case REFUND -> new MetricDefinition("return_cnt_1d", "退货量", "单");
            case CS_TICKET -> new MetricDefinition("cs_ticket_cnt_1d", "咨询工单量", "个");
            case COMPENSATION -> new MetricDefinition("seller_repay_amt_1d", "卖家赔付金额", "元");
            case COUPON -> new MetricDefinition("trade_success_discount_amt_1d", "交易成功优惠金额", "元");
            case GOODS -> new MetricDefinition("goods_apply_cnt_1d", "商品申请量", "个");
            case MERCHANT_OTHER -> new MetricDefinition("appeal_cnt_1d", "申诉次数", "次");
            case SCM -> new MetricDefinition("scm_performance_cnt_1d", "供应链履约量", "单");
            default -> null;
        };
    }

    private Map<String, MetricDefinition> metrics(QuestionCategory category) {
        Map<String, MetricDefinition> map = new LinkedHashMap<>();
        switch (category) {
            case TRADE -> {
                map.put("总gmv|GMV|gmv|成交金额", new MetricDefinition("order_gmv_amt_1d", "总 GMV 金额", "元"));
                map.put("下单用户", new MetricDefinition("order_user_cnt_1d", "下单用户量", "人"));
                map.put("总订单|订单量", new MetricDefinition("order_cnt_1d", "总订单量", "单"));
                map.put("交易成功订单|成交订单", new MetricDefinition("trade_success_order_cnt_1d", "交易成功订单量", "单"));
                map.put("交易成功gmv|交易成功金额", new MetricDefinition("trade_success_gmv_amt_1d", "交易成功 GMV 金额", "元"));
                map.put("客单价", new MetricDefinition("avg_pay_order_amt_1d", "支付成功客单价", "元"));
                map.put("发货超时", new MetricDefinition("ship_timeout_order_cnt_1d", "发货超时订单量", "单"));
            }
            case REFUND -> {
                map.put("退货成功金额", new MetricDefinition("return_success_amt_1d", "退货成功金额", "元"));
                map.put("退款金额", new MetricDefinition("refund_amt_1d", "退款金额", "元"));
                map.put("退货成功", new MetricDefinition("return_success_cnt_1d", "退货成功量", "单"));
                map.put("退货量|退货", new MetricDefinition("return_cnt_1d", "退货量", "单"));
                map.put("直接退款", new MetricDefinition("direct_refund_cnt_1d", "直接退款量", "单"));
                map.put("退款率|退货率", new MetricDefinition("refund_rate_1d", "退货量占支付订单量比例", "%"));
            }
            case CS_TICKET -> {
                map.put("咨询工单|工单量|客服工单", new MetricDefinition("cs_ticket_cnt_1d", "咨询工单量", "个"));
                map.put("二次开启", new MetricDefinition("ticket_reopen_cnt_1d", "工单二次开启量", "个"));
                map.put("催单", new MetricDefinition("ticket_reminder_cnt_1d", "催单工单量", "个"));
                map.put("关闭工单", new MetricDefinition("ticket_close_cnt_1d", "关闭工单量", "个"));
                map.put("评价分", new MetricDefinition("avg_ticket_score_1d", "平均工单评价分", "分"));
            }
            case COMPENSATION -> {
                map.put("赔付金额|赔款金额", new MetricDefinition("seller_repay_amt_1d", "卖家赔付金额", "元"));
                map.put("赔付订单|赔付单", new MetricDefinition("seller_repay_order_cnt_1d", "卖家赔付订单数", "单"));
            }
            case COUPON -> {
                map.put("支付成功优惠单", new MetricDefinition("pay_success_discount_order_cnt_1d", "支付成功优惠单量", "单"));
                map.put("支付成功优惠金额", new MetricDefinition("pay_success_discount_amt_1d", "支付成功优惠金额", "元"));
                map.put("交易成功优惠单", new MetricDefinition("trade_success_discount_order_cnt_1d", "交易成功优惠单量", "单"));
                map.put("交易成功优惠金额|优惠金额", new MetricDefinition("trade_success_discount_amt_1d", "交易成功优惠金额", "元"));
                map.put("优惠占比|优惠比例", new MetricDefinition("pay_discount_rate_1d", "支付成功优惠金额占支付 GMV 比例", "%"));
            }
            case GOODS -> {
                map.put("审核拒绝|拒绝量", new MetricDefinition("goods_audit_reject_cnt_1d", "商品审核拒绝量", "个"));
                map.put("审核通过", new MetricDefinition("goods_audit_pass_cnt_1d", "商品审核通过量", "个"));
                map.put("上架商品|上架量", new MetricDefinition("goods_online_cnt_1d", "上架商品量", "个"));
                map.put("商品申请|申请量", new MetricDefinition("goods_apply_cnt_1d", "商品申请量", "个"));
            }
            case MERCHANT_OTHER -> {
                map.put("保证金缴纳|缴纳保证金", new MetricDefinition("deposit_pay_cnt_1d", "缴纳保证金次数", "次"));
                map.put("申诉成功", new MetricDefinition("appeal_success_cnt_1d", "申诉成功次数", "次"));
                map.put("申诉", new MetricDefinition("appeal_cnt_1d", "申诉次数", "次"));
                map.put("处罚", new MetricDefinition("punish_cnt_1d", "处罚次数", "次"));
            }
            case SCM -> map.put("履约|供应链", new MetricDefinition("scm_performance_cnt_1d", "供应链履约量", "单"));
            default -> {
            }
        }
        return map;
    }

    private QuestionIntent maybeAnalyzeWithLlm(
            String question,
            QuestionIntent intent,
            List<Map<String, Object>> historyRows,
            String wiki
    ) {
        if (!shouldUseLlm(question, intent)) {
            intent.setAnalysisNote("规则意图识别命中");
            return intent;
        }
        return llmIntentAnalysisService.refine(question, intent, historyRows, wiki);
    }

    private QueryPlan buildRulePlan(String question) {
        QueryPlan plan = new QueryPlan();
        if (isRuleGuidanceQuestion(question)) {
            plan.getIntents().add(recognizeByRule(question));
            return plan;
        }
        List<String> segments = splitQuestionSegments(question);
        if (segments.size() <= 1) {
            plan.getIntents().add(recognizeByRule(question));
            return plan;
        }

        Set<String> seen = new LinkedHashSet<>();
        for (String segment : segments) {
            QuestionIntent intent = recognizeByRule(segment);
            String signature = signature(intent);
            if (seen.add(signature)) {
                plan.getIntents().add(intent);
            }
        }
        dropInvalidIntentsWhenPlanHasValidIntent(plan);
        if (plan.getIntents().isEmpty()) {
            plan.getIntents().add(recognizeByRule(question));
        }
        return plan;
    }

    private QueryPlan normalizePlanTimeRange(String question, QueryPlan plan) {
        if (plan == null || plan.getIntents().isEmpty() || !hasExplicitTimeRange(question)) {
            return plan;
        }
        int days = extractDays(question);
        for (QuestionIntent intent : plan.getIntents()) {
            if (intent.getIntentType() == IntentType.VALID && !hasExplicitTimeRange(intent.getQuestion())) {
                intent.setDays(days);
            }
        }
        return plan;
    }

    private boolean shouldUseLlmForPlan(String question, QueryPlan fallbackPlan) {
        if (question == null || question.isBlank()) {
            return false;
        }
        if (isRuleGuidanceQuestion(question)) {
            return false;
        }
        if (!llmIntentAnalysisService.isAvailable()) {
            return false;
        }
        if (matchedBusinessCategoryCount(question) >= 2) {
            return true;
        }
        if (fallbackPlan.getIntents().size() > 1) {
            return true;
        }
        return containsAny(question,
                "分析", "原因", "影响", "关联", "综合", "分别", "同时", "并且",
                "一起", "对比", "环比", "同比", "哪些问题", "为什么", "怎么看");
    }

    private boolean shouldUseLlm(String question, QuestionIntent intent) {
        if (question == null || question.isBlank() || intent.getIntentType() == IntentType.GREETING) {
            return false;
        }
        if (isRuleGuidanceQuestion(question)) {
            return false;
        }
        if (intent.getIntentType() == IntentType.INVALID) {
            return true;
        }
        if (matchedBusinessCategoryCount(question) >= 2) {
            return true;
        }
        if (containsAny(question,
                "为什么", "原因", "建议", "分析", "异常", "波动", "下降", "下滑", "上涨", "增长",
                "对比", "环比", "同比", "影响", "风险", "优化", "怎么办", "怎么处理", "怎么提升",
                "帮我看看", "帮我看下", "帮我分析", "综合", "同时", "分别", "并且", "哪些问题")) {
            return true;
        }
        return question.length() >= 28 && containsAny(question, "看", "查", "咨询", "情况", "数据", "指标", "明细");
    }

    private int matchedBusinessCategoryCount(String question) {
        int count = 0;
        if (containsAny(question, "规则", "政策", "培训", "平台要求")) count++;
        if (containsAny(question, "交易", "订单", "gmv", "GMV", "支付", "成交", "下单", "客单价", "签收", "发货超时")) count++;
        if (containsAny(question, "退货", "退款", "退换", "直接退款", "卖家责任")) count++;
        if (containsAny(question, "客服", "工单", "咨询", "催单", "评价分", "二次开启")) count++;
        if (containsAny(question, "理赔", "赔付", "赔款", "补偿")) count++;
        if (containsAny(question, "优惠券", "优惠", "券", "补贴", "折扣")) count++;
        if (containsAny(question, "商品", "上架", "审核拒绝", "审核通过", "货品", "spu", "SKU", "sku")) count++;
        if (containsAny(question, "保证金", "申诉", "处罚", "入驻", "商家信息", "费率", "结算")) count++;
        if (containsAny(question, "身份", "联系人", "营业执照", "银行卡", "商户类型", "发票", "地址")) count++;
        if (containsAny(question, "供应链", "入库", "出库", "质检", "鉴定", "仓库", "履约", "分拣")) count++;
        return count;
    }

    private boolean isRuleGuidanceQuestion(String question) {
        if (question == null || question.isBlank()) {
            return false;
        }
        boolean asksRule = containsAny(question,
                "规则", "政策", "流程", "要求", "规范", "怎么上架", "如何上架", "怎么做", "怎么办", "有吗",
                "资质", "类目", "图文", "主图", "详情页", "审核被拒", "拒绝原因", "怎么选", "怎么提交");
        if (!asksRule) {
            return false;
        }
        boolean hasBusinessScenario = containsAny(question,
                "商品", "上架", "货品", "订单", "发货", "退货", "退款", "工单", "客服", "赔付", "理赔", "优惠券", "保证金",
                "品牌资质", "资质要求", "需要哪些资质", "类目规范", "类目要求", "商品类目", "商品图片", "图文描述");
        if (!hasBusinessScenario) {
            return false;
        }
        return !containsAny(question,
                "最近", "近", "昨天", "昨日", "多少", "数据", "趋势", "明细", "记录", "金额", "统计", "占比", "量");
    }

    private RuleTopic resolveRuleTopic(String question) {
        if (containsAny(question, "类目", "商品类目", "类目规范", "类目要求", "类目怎么选", "放错类目")) {
            return RuleTopic.GOODS_CATEGORY;
        }
        if (containsAny(question, "资质", "资质要求", "需要哪些资质", "品牌资质", "授权资质", "经营资质")) {
            return RuleTopic.GOODS_QUALIFICATION;
        }
        if (containsAny(question, "图文要求", "主图要求", "详情页要求", "图片要求", "描述要求", "图文描述")) {
            return RuleTopic.GOODS_CONTENT;
        }
        if (containsAny(question, "审核被拒", "拒绝原因", "审核拒绝", "被驳回", "重新提交")) {
            return RuleTopic.GOODS_REJECT;
        }
        if (containsAny(question, "商品", "上架", "货品", "spu", "SPU", "sku", "SKU", "审核")) {
            return RuleTopic.GOODS_LISTING;
        }
        if (containsAny(question, "发货", "签收", "物流", "超时", "订单")) {
            return RuleTopic.DELIVERY;
        }
        if (containsAny(question, "退货", "退款", "退换", "售后")) {
            return RuleTopic.REFUND;
        }
        if (containsAny(question, "客服", "工单", "咨询", "催单")) {
            return RuleTopic.CS_TICKET;
        }
        if (containsAny(question, "理赔", "赔付", "赔款", "补偿")) {
            return RuleTopic.COMPENSATION;
        }
        if (containsAny(question, "优惠券", "优惠", "券", "补贴", "折扣")) {
            return RuleTopic.COUPON;
        }
        if (containsAny(question, "保证金", "缴纳", "充值", "冻结")) {
            return RuleTopic.DEPOSIT;
        }
        return RuleTopic.GENERAL;
    }

    private boolean containsAny(String text, String... keywords) {
        for (String keyword : keywords) {
            if (keyword != null && !keyword.isBlank() && text.contains(keyword)) {
                return true;
            }
        }
        return false;
    }

    private List<String> splitQuestionSegments(String question) {
        List<String> segments = new ArrayList<>();
        if (!containsAny(question, "同时", "并且", "而且", "另外", "还有", "分别", "以及", "及", "和", "；", ";", "，", ",")) {
            return List.of(question);
        }
        String normalized = question
                .replace("；", "|")
                .replace(";", "|")
                .replace("。", "|")
                .replace("同时", "|")
                .replace("并且", "|")
                .replace("而且", "|")
                .replace("另外", "|")
                .replace("还有", "|")
                .replace("分别", "|")
                .replace("以及", "|")
                .replace("，", "|")
                .replace(",", "|");
        for (String segment : normalized.split("\\|")) {
            String candidate = segment == null ? "" : segment.trim();
            if (candidate.isBlank() || isQuestionFillerSegment(candidate)) {
                continue;
            }
            List<String> conjunctionSegments = splitByPairConjunction(candidate);
            if (conjunctionSegments.size() > 1) {
                segments.addAll(conjunctionSegments);
            } else {
                segments.add(candidate);
            }
        }
        if (segments.size() == 1 && containsAny(question, "和", "及")) {
            List<String> conjunctionSegments = splitByPairConjunction(question);
            if (conjunctionSegments.size() > 1) {
                return conjunctionSegments;
            }
        }
        return segments.isEmpty() ? List.of(question) : segments;
    }

    private List<String> splitByPairConjunction(String question) {
        List<String> segments = new ArrayList<>();
        for (String conjunction : List.of("和", "及")) {
            int index = question.indexOf(conjunction);
            if (index <= 0 || index >= question.length() - 1) {
                continue;
            }
            String left = question.substring(0, index).trim();
            String right = question.substring(index + conjunction.length()).trim();
            if (left.isBlank() || right.isBlank()) {
                continue;
            }
            String timePrefix = leadingTimePrefix(left);
            if (!timePrefix.isBlank() && leadingTimePrefix(right).isBlank()) {
                right = timePrefix + right;
            }
            QuestionCategory leftCategory = detectCategory(left);
            QuestionCategory rightCategory = detectCategory(right);
            if (leftCategory != QuestionCategory.UNKNOWN
                    && rightCategory != QuestionCategory.UNKNOWN
                    && leftCategory != rightCategory) {
                segments.add(left);
                segments.add(right);
                return segments;
            }
        }
        return List.of(question);
    }

    private boolean isQuestionFillerSegment(String segment) {
        String compact = segment.replaceAll("[?？。!！,，\\s]", "");
        return compact.isBlank()
                || compact.equals("是多少")
                || compact.equals("多少")
                || compact.equals("什么")
                || compact.equals("看一下")
                || compact.equals("查一下")
                || compact.equals("分别是多少");
    }

    private String leadingTimePrefix(String text) {
        if (text == null) {
            return "";
        }
        String trimmed = text.trim();
        Matcher dayMatcher = RECENT_DAYS.matcher(trimmed);
        if (dayMatcher.find() && dayMatcher.start() == 0) {
            return dayMatcher.group();
        }
        Matcher weekMatcher = RECENT_WEEKS.matcher(trimmed);
        if (weekMatcher.find() && weekMatcher.start() == 0) {
            return weekMatcher.group();
        }
        for (String prefix : List.of("昨天", "昨日", "近一周", "最近一周", "过去一周", "近半个月", "最近半个月", "过去半个月", "近一个月", "最近一个月", "过去一个月")) {
            if (trimmed.startsWith(prefix)) {
                return prefix;
            }
        }
        return "";
    }

    private void dropInvalidIntentsWhenPlanHasValidIntent(QueryPlan plan) {
        boolean hasValidIntent = plan.getIntents().stream()
                .anyMatch(intent -> intent.getIntentType() == IntentType.VALID);
        if (hasValidIntent) {
            plan.getIntents().removeIf(intent -> intent.getIntentType() != IntentType.VALID);
        }
    }

    private String signature(QuestionIntent intent) {
        return "%s|%s|%s|%s|%d".formatted(
                intent.getIntentType(),
                intent.getCategory(),
                intent.getAnswerMode(),
                intent.getMetricColumn() + ":" + intent.getIdentityColumn() + ":" + intent.getRuleTopic(),
                intent.getDays()
        );
    }
}
