package com.yshopping.merchantai.config;

import javax.sql.DataSource;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.datasource.DriverManagerDataSource;

@Configuration
@EnableConfigurationProperties(AppProperties.class)
/**
 * 数据源配置。
 *
 * <p>Doris 用于经营数据分析查询；answer 数据源用于保存商家问答记录。
 * 本地默认都可指向 Doris 9030，生产可将 answer 切到独立 MySQL。</p>
 */
public class DataSourceConfig {

    @Bean
    public DataSource dorisDataSource(AppProperties properties) {
        return dataSource(properties.getDatasource().getDoris());
    }

    @Bean
    public DataSource answerDataSource(AppProperties properties) {
        return dataSource(properties.getDatasource().getAnswer());
    }

    @Bean
    public JdbcTemplate dorisJdbcTemplate(@Qualifier("dorisDataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }

    @Bean
    public JdbcTemplate answerJdbcTemplate(@Qualifier("answerDataSource") DataSource dataSource) {
        return new JdbcTemplate(dataSource);
    }

    private DataSource dataSource(AppProperties.Db db) {
        DriverManagerDataSource dataSource = new DriverManagerDataSource();
        dataSource.setDriverClassName("com.mysql.cj.jdbc.Driver");
        dataSource.setUrl(db.getUrl());
        dataSource.setUsername(db.getUsername());
        dataSource.setPassword(db.getPassword());
        return dataSource;
    }
}
