package com.yshopping.merchantai.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "yshopping")
/**
 * yshopping 应用配置映射。
 *
 * <p>集中承载公司名、默认商家 id、LLM 网关配置、wiki 路径以及 Doris/问答库
 * 两套数据源配置，避免业务代码里散落环境变量读取逻辑。</p>
 */
public class AppProperties {
    private String companyName = "yshopping";
    private String merchantId = "100";
    private String wikiPath = "";
    private final Llm llm = new Llm();
    private final Datasource datasource = new Datasource();

    public String getCompanyName() {
        return companyName;
    }

    public void setCompanyName(String companyName) {
        this.companyName = companyName;
    }

    public String getMerchantId() {
        return merchantId;
    }

    public void setMerchantId(String merchantId) {
        this.merchantId = merchantId;
    }

    public String getWikiPath() {
        return wikiPath;
    }

    public void setWikiPath(String wikiPath) {
        this.wikiPath = wikiPath;
    }

    public Llm getLlm() {
        return llm;
    }

    public Datasource getDatasource() {
        return datasource;
    }

    public static class Llm {
        private String baseUrl = "https://way.ydata.vip/v1";
        private String model = "gpt-5.5";
        private String apiKey = "";

        public String getBaseUrl() {
            return baseUrl;
        }

        public void setBaseUrl(String baseUrl) {
            this.baseUrl = baseUrl;
        }

        public String getModel() {
            return model;
        }

        public void setModel(String model) {
            this.model = model;
        }

        public String getApiKey() {
            return apiKey;
        }

        public void setApiKey(String apiKey) {
            this.apiKey = apiKey;
        }
    }

    public static class Datasource {
        private final Db doris = new Db();
        private final Db answer = new Db();

        public Db getDoris() {
            return doris;
        }

        public Db getAnswer() {
            return answer;
        }
    }

    public static class Db {
        private String url;
        private String username;
        private String password = "";

        public String getUrl() {
            return url;
        }

        public void setUrl(String url) {
            this.url = url;
        }

        public String getUsername() {
            return username;
        }

        public void setUsername(String username) {
            this.username = username;
        }

        public String getPassword() {
            return password;
        }

        public void setPassword(String password) {
            this.password = password;
        }
    }
}
