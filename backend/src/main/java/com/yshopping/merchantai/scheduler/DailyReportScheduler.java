package com.yshopping.merchantai.scheduler;

import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.dto.DailyReportResponse;
import com.yshopping.merchantai.service.DailyReportService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Component;

@Component
/**
 * 每日经营日报定时器。
 *
 * <p>每天 10 点读取昨日商家经营画像，当前先打印日志，后续可替换为站内信、
 * IM 或商家助手消息推送。</p>
 */
public class DailyReportScheduler {
    private static final Logger log = LoggerFactory.getLogger(DailyReportScheduler.class);
    private final AppProperties properties;
    private final DailyReportService dailyReportService;

    public DailyReportScheduler(AppProperties properties, DailyReportService dailyReportService) {
        this.properties = properties;
        this.dailyReportService = dailyReportService;
    }

    @Scheduled(cron = "0 0 10 * * *", zone = "Asia/Shanghai")
    public void pushDailyReport() {
        DailyReportResponse report = dailyReportService.report(properties.getMerchantId());
        // 这里保留为商家助手推送入口：接入 IM/站内信后替换为真实发送。
        log.info("yshopping 商家助手每日 10 点经营推送：merchantId={}, date={}, metrics={}, suggestions={}",
                report.getMerchantId(), report.getDate(), report.getMetrics(), report.getSuggestions());
    }
}
