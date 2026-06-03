package com.yshopping.merchantai.repository;

import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Repository;

@Repository
/**
 * Doris 数据访问封装。
 *
 * <p>所有经营画像、明细表和商家维表查询都通过这里执行，便于后续统一加 SQL 审计、
 * 超时控制或查询日志。</p>
 */
public class DorisRepository {
    private final JdbcTemplate jdbcTemplate;

    public DorisRepository(@Qualifier("dorisJdbcTemplate") JdbcTemplate jdbcTemplate) {
        this.jdbcTemplate = jdbcTemplate;
    }

    public List<Map<String, Object>> query(String sql, Object... args) {
        return jdbcTemplate.queryForList(sql, args);
    }

    /**
     * 查询单行结果；无数据时返回空 Map，避免调用方处理 EmptyResultDataAccessException。
     */
    public Map<String, Object> queryOne(String sql, Object... args) {
        List<Map<String, Object>> rows = query(sql, args);
        return rows.isEmpty() ? Map.of() : rows.get(0);
    }
}
