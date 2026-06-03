package com.yshopping.merchantai.service;

import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.MerchantInfo;
import com.yshopping.merchantai.repository.DorisRepository;
import java.util.Map;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * 商家识别服务。
 *
 * <p>优先从 Doris 商家维表 dim_merchant_df 读取当前商家资料；
 * 如果维表不可用，则返回默认商家信息保证演示链路不中断。</p>
 */
public class MerchantService {
    private final AppProperties properties;
    private final DorisRepository dorisRepository;

    public MerchantService(AppProperties properties, DorisRepository dorisRepository) {
        this.properties = properties;
        this.dorisRepository = dorisRepository;
    }

    public MerchantInfo currentMerchant(String requestedMerchantId) {
        String merchantId = StringUtils.hasText(requestedMerchantId) ? requestedMerchantId : properties.getMerchantId();
        try {
            Map<String, Object> row = dorisRepository.queryOne(
                    """
                    SELECT merchant_id, user_id, company_name, merchant_type_name
                    FROM dim_merchant_df
                    WHERE merchant_id = ?
                    ORDER BY pt DESC
                    LIMIT 1
                    """,
                    merchantId
            );
            if (!row.isEmpty()) {
                String companyName = stringValue(row.get("company_name"), "yshopping商家");
                return new MerchantInfo(
                        stringValue(row.get("merchant_id"), merchantId),
                        stringValue(row.get("user_id"), ""),
                        companyName,
                        companyName
                );
            }
        } catch (Exception ignored) {
            // Doris 维表不可用时仍允许本地演示。
        }
        return new MerchantInfo(merchantId, "", "yshopping商家" + merchantId, "yshopping商家" + merchantId);
    }

    private String stringValue(Object value, String fallback) {
        if (value == null) {
            return fallback;
        }
        String text = String.valueOf(value);
        return text.isBlank() ? fallback : text;
    }
}
