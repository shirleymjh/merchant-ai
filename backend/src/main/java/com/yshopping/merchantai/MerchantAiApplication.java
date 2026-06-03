package com.yshopping.merchantai;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@EnableScheduling
@SpringBootApplication
/**
 * yshopping 商家 AI 助手后端启动类。
 *
 * <p>开启 Spring Boot 自动装配和定时任务，用于承载问答接口、Doris 查询、
 * 问答记录落库以及每日 10 点经营日报推送。</p>
 */
public class MerchantAiApplication {
    public static void main(String[] args) {
        SpringApplication.run(MerchantAiApplication.class, args);
    }
}
