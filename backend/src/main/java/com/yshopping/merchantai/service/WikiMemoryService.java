package com.yshopping.merchantai.service;

import com.yshopping.merchantai.config.AppProperties;
import com.yshopping.merchantai.model.QuestionCategory;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.Resource;
import org.springframework.core.io.support.PathMatchingResourcePatternResolver;
import org.springframework.stereotype.Service;
import org.springframework.util.StringUtils;

@Service
/**
 * LLM wiki 记忆服务。
 *
 * <p>读取内置 rule.md 和 merchant_qa_memory.md，也支持把历史问答压缩沉淀到 runtime/wiki。</p>
 */
public class WikiMemoryService {
    private static final Logger log = LoggerFactory.getLogger(WikiMemoryService.class);
    private final AppProperties properties;
    private final LlmClient llmClient;

    public WikiMemoryService(AppProperties properties, LlmClient llmClient) {
        this.properties = properties;
        this.llmClient = llmClient;
    }

    public String loadRelevantWiki(QuestionCategory category) {
        return loadRelevantWiki(List.of(category));
    }

    public String loadRelevantWiki(List<QuestionCategory> categories) {
        StringBuilder builder = new StringBuilder();
        builder.append(loadClasspathWiki());
        builder.append("\n\n");
        builder.append(loadRuntimeWiki(categories));
        return truncate(builder.toString(), 16_000);
    }

    public String loadBaseWiki() {
        StringBuilder builder = new StringBuilder();
        builder.append(loadClasspathWiki());
        builder.append("\n\n");
        builder.append(loadRuntimeWiki(List.of(QuestionCategory.UNKNOWN)));
        return truncate(builder.toString(), 8_000);
    }

    public List<String> suggestedQuestions(QuestionCategory category, int limit) {
        if (category == null || category == QuestionCategory.UNKNOWN || limit <= 0) {
            return List.of();
        }
        Path wikiDir = runtimeWikiDir();
        if (!Files.isDirectory(wikiDir)) {
            return List.of();
        }
        Set<String> suggestions = new LinkedHashSet<>();
        try (Stream<Path> stream = Files.list(wikiDir)) {
            List<Path> files = stream
                    .filter(path -> path.getFileName().toString().endsWith(".md"))
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
            for (Path file : files) {
                String name = file.getFileName().toString();
                if (!matchesCategory(name, List.of(category))) {
                    continue;
                }
                suggestions.addAll(parseSuggestedQuestions(Files.readString(file, StandardCharsets.UTF_8)));
                if (suggestions.size() >= limit) {
                    break;
                }
            }
        } catch (IOException e) {
            return List.of();
        }
        return suggestions.stream().limit(limit).toList();
    }

    public String platformRules() {
        try {
            Resource resource = new PathMatchingResourcePatternResolver().getResource("classpath:wiki/rule.md");
            return resource.exists() ? resource.getContentAsString(StandardCharsets.UTF_8) : "";
        } catch (IOException e) {
            return "";
        }
    }

    public Path compressToWiki(String categoryName, List<?> historyRows, String manualMarkdown) {
        try {
            Path wikiDir = runtimeWikiDir();
            Files.createDirectories(wikiDir);
            String safeName = sanitizeFileName(StringUtils.hasText(categoryName) ? categoryName : "通用");
            Path output = wikiDir.resolve(safeName + ".md");
            String prompt = """
                    请把以下 yshopping 商家 AI 助手历史问答压缩成可复用的 wiki 记忆。
                    要求：
                    1. 按业务分类沉淀用户意图、可用表、字段、口径和推荐回复话术。
                    2. 信息要短、准、可复用。
                    3. 如果人工补充内容存在，优先保留人工补充。
                    4. 结尾必须补充一个“## 推荐问法”小节，并且只输出 3 条推荐问法。
                    5. 推荐问法要和当前分类强相关，优先选择用户最可能继续追问的指标、趋势或明细问题。
                    6. 输出使用 Markdown，推荐问法必须使用 `- ` 列表格式。

                    人工补充：
                    %s

                    历史问答：
                    %s
                    """.formatted(manualMarkdown == null ? "" : manualMarkdown, historyRows);
            String fallback = "# " + categoryName + "\n\n" + (manualMarkdown == null ? "" : manualMarkdown)
                    + "\n\n## 推荐问法\n"
                    + "- 查看" + categoryName + "概况\n"
                    + "- 最近7天" + categoryName + "趋势\n"
                    + "- 查看" + categoryName + "明细\n"
                    + "\n更新时间：" + LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")) + "\n";
            String content = llmClient.chat("你是 yshopping 商家 AI 助手的知识库整理员。", prompt, fallback);
            Files.writeString(output, content, StandardCharsets.UTF_8);
            return output;
        } catch (IOException e) {
            throw new IllegalStateException("写入 wiki 失败", e);
        }
    }

    public Map<String, Path> compressMerchantWiki(String merchantId, List<QuestionCategory> categories, WikiHistoryLoader historyLoader) {
        Map<String, Path> outputs = new LinkedHashMap<>();
        for (QuestionCategory category : categories) {
            if (category == null || category == QuestionCategory.UNKNOWN) {
                continue;
            }
            List<?> historyRows = historyLoader.load(merchantId, category.getDisplayName());
            if (historyRows == null || historyRows.isEmpty()) {
                continue;
            }
            Path path = compressToWiki(category.getDisplayName(), historyRows, "");
            outputs.put(category.getDisplayName(), path);
        }
        return outputs;
    }

    private String loadClasspathWiki() {
        StringBuilder builder = new StringBuilder();
        try {
            Resource[] resources = new PathMatchingResourcePatternResolver().getResources("classpath:wiki/*.md");
            for (Resource resource : resources) {
                builder.append("\n\n## ").append(resource.getFilename()).append("\n");
                builder.append(resource.getContentAsString(StandardCharsets.UTF_8));
            }
        } catch (IOException e) {
            log.warn("读取内置 wiki 失败", e);
        }
        return builder.toString();
    }

    private String loadRuntimeWiki(List<QuestionCategory> categories) {
        Path wikiDir = runtimeWikiDir();
        if (!Files.isDirectory(wikiDir)) {
            return "";
        }
        try (Stream<Path> stream = Files.list(wikiDir)) {
            List<Path> files = stream
                    .filter(path -> path.getFileName().toString().endsWith(".md"))
                    .sorted(Comparator.comparing(Path::toString))
                    .toList();
            StringBuilder builder = new StringBuilder();
            for (Path file : files) {
                String name = file.getFileName().toString();
                if (matchesCategory(name, categories)) {
                    builder.append("\n\n## ").append(name).append("\n");
                    builder.append(Files.readString(file, StandardCharsets.UTF_8));
                }
            }
            return builder.toString();
        } catch (IOException e) {
            return "";
        }
    }

    private Path runtimeWikiDir() {
        if (StringUtils.hasText(properties.getWikiPath())) {
            return Path.of(properties.getWikiPath());
        }
        return Path.of("runtime", "wiki");
    }

    private String truncate(String text, int maxLength) {
        if (text == null || text.length() <= maxLength) {
            return text == null ? "" : text;
        }
        return text.substring(0, maxLength);
    }

    private String sanitizeFileName(String name) {
        return name.replaceAll("[\\\\/:*?\"<>|\\s]+", "_");
    }

    private boolean matchesCategory(String fileName, List<QuestionCategory> categories) {
        if (categories == null || categories.isEmpty()) {
            return fileName.contains("通用");
        }
        if (fileName.contains("通用")) {
            return true;
        }
        for (QuestionCategory category : categories) {
            if (category == QuestionCategory.UNKNOWN || fileName.contains(category.getDisplayName())) {
                return true;
            }
        }
        return false;
    }

    private List<String> parseSuggestedQuestions(String markdown) {
        if (!StringUtils.hasText(markdown)) {
            return List.of();
        }
        String[] lines = markdown.split("\\R");
        List<String> results = new java.util.ArrayList<>();
        boolean inSection = false;
        for (String rawLine : lines) {
            String line = rawLine == null ? "" : rawLine.trim();
            if (line.startsWith("## ")) {
                inSection = line.equals("## 推荐问法");
                continue;
            }
            if (!inSection) {
                continue;
            }
            if (line.startsWith("- ") || line.startsWith("* ")) {
                String question = line.substring(2).trim();
                if (!question.isBlank()) {
                    results.add(question);
                }
            } else if (StringUtils.hasText(line)) {
                break;
            }
        }
        return results;
    }

    @FunctionalInterface
    public interface WikiHistoryLoader {
        List<?> load(String merchantId, String categoryName);
    }
}
