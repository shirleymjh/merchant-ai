package com.yshopping.merchantai.model;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Doris 查询结果包。
 *
 * <p>同时记录用到的表、执行 SQL 和返回数据，方便 LLM 组织回答和前端展示数据来源。</p>
 */
public class QueryBundle {
    private final List<String> tables = new ArrayList<>();
    private final List<String> sqls = new ArrayList<>();
    private final List<Map<String, Object>> rows = new ArrayList<>();

    public List<String> getTables() {
        return tables;
    }

    public List<String> getSqls() {
        return sqls;
    }

    public List<Map<String, Object>> getRows() {
        return rows;
    }
}
