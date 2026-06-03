package com.yshopping.merchantai.service;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.yshopping.merchantai.config.AppProperties;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * OpenAI 兼容 Chat Completions 客户端。
 *
 * <p>通过 base_url、model、api_key 调用 gpt-5.5；未配置密钥或请求失败时返回降级话术，
 * 保证页面和 Doris 查询链路仍可演示。</p>
 */
public class LlmClient {
    private static final Logger log = LoggerFactory.getLogger(LlmClient.class);
    private final AppProperties properties;
    private final ObjectMapper objectMapper;
    private final HttpClient httpClient;

    public LlmClient(AppProperties properties, ObjectMapper objectMapper) {
        this.properties = properties;
        this.objectMapper = objectMapper;
        this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build();
    }

    public boolean isConfigured() {
        return StringUtils.hasText(properties.getLlm().getApiKey());
    }

    public String chat(String systemPrompt, String userPrompt, String fallback) {
        if (!isConfigured()) {
            return fallback;
        }
        try {
            Map<String, Object> requestBody = Map.of(
                    "model", properties.getLlm().getModel(),
                    "messages", List.of(
                            Map.of("role", "system", "content", systemPrompt),
                            Map.of("role", "user", "content", userPrompt)
                    ),
                    "temperature", 0.2
            );
            String body = objectMapper.writeValueAsString(requestBody);
            String baseUrl = properties.getLlm().getBaseUrl().replaceAll("/+$", "");
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(baseUrl + "/chat/completions"))
                    .timeout(Duration.ofSeconds(45))
                    .header("Authorization", "Bearer " + properties.getLlm().getApiKey())
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                    .build();
            HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
            if (response.statusCode() < 200 || response.statusCode() >= 300) {
                log.warn("LLM 请求失败，status={} body={}", response.statusCode(), response.body());
                return fallback;
            }
            JsonNode root = objectMapper.readTree(response.body());
            JsonNode content = root.path("choices").path(0).path("message").path("content");
            return content.isMissingNode() ? fallback : content.asText(fallback);
        } catch (Exception e) {
            log.warn("LLM 请求异常，使用降级回复", e);
            return fallback;
        }
    }
}
