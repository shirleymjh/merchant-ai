package com.yshopping.merchantai.model;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/**
 * 用户问题的查询计划。
 *
 * <p>一个自然语言问题可能会拆成多个结构化意图，图执行阶段按顺序逐个处理。</p>
 */
public class QueryPlan {
    private final List<QuestionIntent> intents = new ArrayList<>();

    public List<QuestionIntent> getIntents() {
        return intents;
    }

    public QuestionIntent primaryIntent() {
        return intents.isEmpty() ? new QuestionIntent() : intents.get(0);
    }

    public List<QuestionCategory> categories() {
        Set<QuestionCategory> categories = new LinkedHashSet<>();
        for (QuestionIntent intent : intents) {
            categories.add(intent.getCategory());
        }
        return new ArrayList<>(categories);
    }
}
