package com.yshopping.merchantai.model;

/**
 * 用户问题解析后的结构化意图。
 *
 * <p>LangGraph 的后续节点只读取这个对象，不再直接解析原始自然语言。</p>
 */
public class QuestionIntent {
    private IntentType intentType = IntentType.INVALID;
    private QuestionCategory category = QuestionCategory.UNKNOWN;
    private AnswerMode answerMode = AnswerMode.INVALID;
    private String question = "";
    private String metricColumn = "";
    private String metricName = "";
    private String metricUnit = "";
    private String identityColumn = "";
    private String identityName = "";
    private RuleTopic ruleTopic = RuleTopic.GENERAL;
    private int days = 1;
    private boolean llmAnalysisRequested;
    private boolean llmAnalyzed;
    private String analysisSource = "RULE";
    private String analysisNote = "";

    public IntentType getIntentType() {
        return intentType;
    }

    public void setIntentType(IntentType intentType) {
        this.intentType = intentType;
    }

    public QuestionCategory getCategory() {
        return category;
    }

    public void setCategory(QuestionCategory category) {
        this.category = category;
    }

    public AnswerMode getAnswerMode() {
        return answerMode;
    }

    public void setAnswerMode(AnswerMode answerMode) {
        this.answerMode = answerMode;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question == null ? "" : question;
    }

    public String getMetricColumn() {
        return metricColumn;
    }

    public void setMetricColumn(String metricColumn) {
        this.metricColumn = metricColumn;
    }

    public String getMetricName() {
        return metricName;
    }

    public void setMetricName(String metricName) {
        this.metricName = metricName;
    }

    public String getMetricUnit() {
        return metricUnit;
    }

    public void setMetricUnit(String metricUnit) {
        this.metricUnit = metricUnit;
    }

    public String getIdentityColumn() {
        return identityColumn;
    }

    public void setIdentityColumn(String identityColumn) {
        this.identityColumn = identityColumn == null ? "" : identityColumn;
    }

    public String getIdentityName() {
        return identityName;
    }

    public void setIdentityName(String identityName) {
        this.identityName = identityName == null ? "" : identityName;
    }

    public RuleTopic getRuleTopic() {
        return ruleTopic;
    }

    public void setRuleTopic(RuleTopic ruleTopic) {
        this.ruleTopic = ruleTopic == null ? RuleTopic.GENERAL : ruleTopic;
    }

    public int getDays() {
        return days;
    }

    public void setDays(int days) {
        this.days = Math.max(1, Math.min(days, 365));
    }

    public boolean isLlmAnalysisRequested() {
        return llmAnalysisRequested;
    }

    public void setLlmAnalysisRequested(boolean llmAnalysisRequested) {
        this.llmAnalysisRequested = llmAnalysisRequested;
    }

    public boolean isLlmAnalyzed() {
        return llmAnalyzed;
    }

    public void setLlmAnalyzed(boolean llmAnalyzed) {
        this.llmAnalyzed = llmAnalyzed;
    }

    public String getAnalysisSource() {
        return analysisSource;
    }

    public void setAnalysisSource(String analysisSource) {
        this.analysisSource = analysisSource == null ? "RULE" : analysisSource;
    }

    public String getAnalysisNote() {
        return analysisNote;
    }

    public void setAnalysisNote(String analysisNote) {
        this.analysisNote = analysisNote == null ? "" : analysisNote;
    }
}
