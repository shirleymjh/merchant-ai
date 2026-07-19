<template>
  <article :id="anchorId || undefined" :class="['message', role]">
    <div v-if="role === 'assistant'" class="assistant-message">
      <div class="assistant-avatar">Y</div>
      <div class="assistant-stream">
        <div class="message-meta">
          <strong>商家经营助手</strong>
          <span>· {{ displayTime }}</span>
        </div>
        <div class="assistant-card">
          <details v-if="steps.length" class="query-runner" open>
            <summary><span>思考完成</span><small>{{ steps.length }} 个步骤</small></summary>
            <div class="thinking">
              <div v-for="step in steps" :key="step" class="thinking-step"><CheckCircle2 :size="14" /><span>{{ step }}</span></div>
            </div>
          </details>
          <div class="answer-text">
            <section
              v-for="(block, blockIndex) in answerBlocks"
              :key="`${blockIndex}-${block.title || block.text}`"
              class="answer-block"
            >
              <p v-if="block.title" class="answer-block-title">{{ block.title }}</p>
              <p v-if="block.text" class="answer-block-text">{{ block.text }}</p>
              <ul v-if="block.items.length" class="answer-block-list">
                <li v-for="item in block.items" :key="item">{{ item }}</li>
              </ul>
            </section>
          </div>
          <section v-if="analysisScopeVisible" class="scope-disclosure">
            <Info :size="15" />
            <div>
              <b>分析范围</b>
              <span>{{ analysisScopeText }}</span>
            </div>
          </section>
          <section v-if="knowledgeProposals.length" class="knowledge-proposals">
            <div class="experience-head">
              <BookOpenCheck :size="16" />
              <h3>待确认的业务知识</h3>
            </div>
            <article v-for="item in knowledgeProposals" :key="item.suggestionId">
              <b>{{ item.title || item.metricName || '业务知识' }}</b>
              <p>{{ item.correctionText || '本轮对话识别到可沉淀的业务规则。' }}</p>
              <span>{{ knowledgeProposalStatusText(item) }}</span>
              <div v-if="item.conflictCheck?.status === 'confirmation_required'" class="knowledge-conflict-card">
                <strong>发现相似或冲突知识</strong>
                <p>{{ item.conflictCheck.message || '保存前需要确认如何处理已有知识。' }}</p>
                <ul>
                  <li v-for="match in (item.conflictCheck.matches || []).slice(0, 3)" :key="match.existingKnowledgeId">
                    {{ match.title }}：{{ match.existingText }}
                  </li>
                </ul>
                <div class="knowledge-proposal-actions">
                  <button
                    v-if="conflictResolutionAllows(item, 'replace')"
                    type="button"
                    :disabled="item.actionPending"
                    @click="handleKnowledgeProposalAction(item, 'accept', 'replace')"
                  >使用新知识</button>
                  <button
                    v-if="conflictResolutionAllows(item, 'merge')"
                    type="button"
                    class="secondary"
                    :disabled="item.actionPending"
                    @click="handleKnowledgeProposalAction(item, 'accept', 'merge')"
                  >融合两条</button>
                  <button
                    v-if="conflictResolutionAllows(item, 'use_existing')"
                    type="button"
                    class="secondary"
                    :disabled="item.actionPending"
                    @click="handleKnowledgeProposalAction(item, 'accept', 'use_existing')"
                  >使用已有知识</button>
                  <button
                    v-if="conflictResolutionAllows(item, 'keep_both')"
                    type="button"
                    class="ghost"
                    :disabled="item.actionPending"
                    @click="handleKnowledgeProposalAction(item, 'accept', 'keep_both')"
                  >保留两条</button>
                  <button
                    v-if="conflictResolutionAllows(item, 'suggest')"
                    type="button"
                    class="secondary"
                    :disabled="item.actionPending"
                    @click="handleKnowledgeProposalAction(item, 'suggest')"
                  >提交平台审核</button>
                  <button type="button" class="ghost" :disabled="item.actionPending" @click="handleKnowledgeProposalAction(item, 'skip')">取消</button>
                </div>
              </div>
              <div v-else-if="knowledgeProposalActionable(item)" class="knowledge-proposal-actions">
                <button
                  v-if="knowledgeProposalAllows(item, 'accept')"
                  type="button"
                  :disabled="item.actionPending"
                  @click="handleKnowledgeProposalAction(item, 'accept')"
                >保存为本店设置</button>
                <button
                  v-if="knowledgeProposalAllows(item, 'suggest')"
                  type="button"
                  class="secondary"
                  :disabled="item.actionPending"
                  @click="handleKnowledgeProposalAction(item, 'suggest')"
                >提交给平台</button>
                <button
                  v-if="knowledgeProposalAllows(item, 'skip')"
                  type="button"
                  class="ghost"
                  :disabled="item.actionPending"
                  @click="handleKnowledgeProposalAction(item, 'skip')"
                >忽略</button>
              </div>
              <small v-if="item.actionError" class="knowledge-proposal-error">{{ item.actionError }}</small>
            </article>
          </section>
          <section v-if="experienceAlerts.length" class="experience-panel alerts">
            <div class="experience-head">
              <AlertTriangle :size="16" />
              <h3>异常提醒</h3>
            </div>
            <div class="alert-list">
              <button
                v-for="alert in experienceAlerts"
                :key="alert.message || alert.metric"
                type="button"
                class="alert-item"
                @click="askFollowUp(alert.drillDownQuestion)"
              >
                <span>{{ alert.metric || '指标' }}</span>
                <strong>{{ alert.message }}</strong>
              </button>
            </div>
          </section>
          <section v-if="!workspaceMode && (businessAdvice.length || drillDownActions.length)" class="experience-panel">
            <div class="experience-head">
              <Sparkles :size="16" />
              <h3>下一步</h3>
            </div>
            <ul v-if="businessAdvice.length" class="business-advice-list">
              <li v-for="item in businessAdvice" :key="item">{{ item }}</li>
            </ul>
            <div v-if="drillDownActions.length" class="drilldown-actions">
              <button
                v-for="action in drillDownActions"
                :key="`${action.label}-${action.question}`"
                type="button"
                @click="askFollowUp(action.question)"
              >
                <span>{{ action.label }}</span>
                <ArrowRight :size="14" />
              </button>
            </div>
          </section>
          <section v-if="traceabilityItems.length || disclosureItems.length" class="evidence-strip">
            <div v-if="traceabilityItems.length" class="evidence-group">
              <Info :size="15" />
              <span v-for="item in traceabilityItems" :key="item">{{ item }}</span>
            </div>
            <div v-if="disclosureItems.length" class="evidence-group">
              <BookOpenCheck :size="15" />
              <span v-for="item in disclosureItems" :key="item">{{ item }}</span>
            </div>
          </section>
          <section v-if="metricDefinitionCards.length" class="metric-definition-panel">
            <div class="experience-head">
              <BookOpenCheck :size="16" />
              <h3>口径确认</h3>
            </div>
            <article v-for="item in metricDefinitionCards" :key="item.metricKey || item.displayName">
              <div>
                <b>{{ item.displayName || item.metricKey || '当前指标' }}</b>
                <p>{{ item.description || '本次使用已发布语义资产中的当前口径。' }}</p>
              </div>
              <div class="metric-definition-actions">
                <button type="button" @click="handleMetricDefinitionAction(item, 'confirm_default')">以后默认这个口径</button>
                <button type="button" class="secondary" @click="handleMetricDefinitionAction(item, 'question')">我对口径有疑问</button>
              </div>
            </article>
          </section>
          <section v-if="clarificationCard" class="confirmation-card">
            <div class="confirmation-card-head">
              <div>
                <p>{{ clarificationCard.stageLabel }}</p>
                <h3>{{ clarificationCard.title }}</h3>
              </div>
              <span>{{ clarificationCard.statusLabel }}</span>
            </div>
            <p class="confirmation-question">{{ clarificationCard.question }}</p>
            <div v-if="clarificationMetricPreview.length || clarificationTablePreview.length" class="confirmation-preview">
              <div v-if="clarificationMetricPreview.length">
                <b>指标预览</b>
                <span>{{ clarificationMetricPreview.join('、') }}</span>
              </div>
              <div v-if="clarificationTablePreview.length">
                <b>数据预览</b>
                <span>{{ clarificationTablePreview.join('、') }}</span>
              </div>
            </div>
            <div v-if="clarificationOptions.length" class="confirmation-options">
              <button
                v-for="option in clarificationOptions"
                :key="option"
                type="button"
                @click="confirmClarification(option)"
              >
                {{ option }}
              </button>
            </div>
          </section>
          <div v-if="metricSummarySections.length" class="metric-summary-grid">
            <section
              v-for="(section, sectionIndex) in metricSummarySections"
              :key="`metric-summary-${sectionIndex}-${section.valueColumn}`"
              class="metric-summary-card"
            >
              <p class="metric-summary-kicker">核心指标</p>
              <h3>{{ section.label }}</h3>
              <strong>{{ section.value }}</strong>
            </section>
          </div>
          <MetricLineChart v-if="!workspaceMode"
            v-for="(section, sectionIndex) in chartSections"
            :key="`chart-${sectionIndex}-${section.title || section.metricName}`"
            :title="section.title || section.metricName"
            :rows="section.rows"
            :tables="section.tables"
          />
          <section
            v-for="(section, sectionIndex) in aggregateSections"
            :key="`aggregate-${sectionIndex}-${section.title || section.tables?.join(',')}`"
            class="aggregate-card"
          >
            <div class="aggregate-card-head">
              <p class="aggregate-card-kicker">{{ section.mode === 'topn' ? '排行结果' : '分组统计' }}</p>
              <h3>{{ presentSectionTitle(section.title) }}</h3>
            </div>
            <div v-if="section.mode === 'topn'" class="ranking-list">
              <div
                v-for="(row, rowIndex) in section.rows"
                :key="`${row.group_value}-${rowIndex}`"
                class="ranking-item"
              >
                <div class="ranking-index">{{ rowIndex + 1 }}</div>
                <div class="ranking-main">
                  <p class="ranking-name">{{ formatAggregateGroup(row) }}</p>
                  <p class="ranking-sub">
                    {{ resolveGroupLabel(section.title) }} · {{ formatAggregateCount(row.sample_count) }}
                  </p>
                </div>
                <div class="ranking-value">{{ formatAggregateMetric(row.metric_value, section.title) }}</div>
              </div>
            </div>
            <table v-else class="detail-table aggregate-table">
              <thead>
                <tr>
                  <th>{{ resolveGroupLabel(section.title) }}</th>
                  <th>{{ resolveMetricLabel(section.title) }}</th>
                  <th>记录数</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIndex) in section.rows" :key="`${row.group_value}-${rowIndex}`">
                  <td>{{ formatAggregateGroup(row) }}</td>
                  <td>{{ formatAggregateMetric(row.metric_value, section.title) }}</td>
                  <td>{{ formatAggregateCount(row.sample_count) }}</td>
                </tr>
              </tbody>
            </table>
          </section>
          <div
            v-for="(section, sectionIndex) in tableSections"
            :key="`table-${sectionIndex}-${section.title || section.tables?.join(',')}`"
            class="detail-table-wrap"
          >
            <div class="detail-table-head">
              <div>
                <p>{{ filteredTableRows(section, sectionIndex).length }} 行</p>
                <h3>{{ presentSectionTitle(section.title) }}</h3>
              </div>
              <div class="result-toolbar" aria-label="表格操作">
                <button type="button" title="放大查看" @click="openExpandedTable(section, sectionIndex)">
                  <Maximize2 :size="14" />
                </button>
                <button type="button" title="下载结果" @click="downloadTable(section, sectionIndex)">
                  <Download :size="14" />
                </button>
                <button
                  type="button"
                  :class="{ active: tableFilterOpen(sectionIndex) }"
                  title="筛选"
                  @click="toggleTableFilter(sectionIndex)"
                >
                  <Filter :size="14" />
                </button>
                <button type="button" title="复制" @click="copyTable(section, sectionIndex)">
                  <Copy :size="14" />
                </button>
              </div>
            </div>
            <div v-if="tableFilterOpen(sectionIndex)" class="table-filter-row">
              <input
                :value="tableFilterQuery(sectionIndex)"
                type="search"
                placeholder="筛选当前表格"
                @input="setTableFilterQuery(sectionIndex, $event.target.value)"
              >
              <button type="button" @click="setTableFilterQuery(sectionIndex, '')">清除</button>
            </div>
            <table class="detail-table">
              <thead>
                <tr>
                  <th v-for="column in section.columns" :key="column" :title="columnLabel(column)">{{ columnLabel(column) }}</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="(row, rowIndex) in filteredTableRows(section, sectionIndex)" :key="rowIndex">
                  <td v-for="column in section.columns" :key="column">{{ formatCell(row[column]) }}</td>
                </tr>
                <tr v-if="!filteredTableRows(section, sectionIndex).length">
                  <td :colspan="section.columns.length" class="empty-table-cell">没有匹配的数据</td>
                </tr>
              </tbody>
            </table>
          </div>
          <button v-if="canBuildReport" type="button" class="analysis-report-entry" @click="reportOpen = true">
            <span class="analysis-report-icon"><FileChartColumnIncreasing :size="20" /></span>
            <span><b>生成经营分析报告</b><small>把本轮结果整理成可汇报、可下载的 HTML</small></span>
            <ArrowRight :size="17" />
          </button>
          <div v-if="id" class="message-actions">
            <button
              type="button"
              :class="['adopt-action', { active: feedbackStatus?.adopted }]"
              title="采纳"
              @click="$emit('feedback', { id, adopted: !feedbackStatus?.adopted })"
            >
              <Check :size="16" />
              <span>{{ feedbackStatus?.adopted ? '已采纳' : '采纳' }}</span>
            </button>
            <button
              type="button"
              :class="{ active: feedbackStatus?.liked }"
              title="点赞"
              @click="$emit('feedback', { id, liked: !feedbackStatus?.liked, disliked: false })"
            >
              <ThumbsUp :size="16" />
            </button>
            <button
              type="button"
              :class="{ active: feedbackStatus?.disliked }"
              title="点踩"
              @click="$emit('feedback', { id, liked: false, disliked: !feedbackStatus?.disliked })"
            >
              <ThumbsDown :size="16" />
            </button>
          </div>
          <p class="ai-note">内容为 AI 生成，仅供参考</p>
        </div>
      </div>
    </div>
    <div v-else class="user-bubble">{{ text }}</div>
    <Teleport v-if="role === 'assistant'" to="body">
      <div v-if="expandedTable" class="result-modal-backdrop" @click.self="closeExpandedTable">
        <section class="result-modal" role="dialog" aria-modal="true" :aria-label="`${expandedTable.title} 放大查看`">
          <div class="result-modal-head">
            <div>
              <p>{{ expandedTable.rows.length }} 行</p>
              <h3>{{ expandedTable.title }}</h3>
            </div>
            <button type="button" title="关闭" @click="closeExpandedTable">
              <X :size="18" />
            </button>
          </div>
          <div class="result-modal-body">
            <div class="result-modal-table-wrap">
              <table class="detail-table result-modal-table">
                <thead>
                  <tr>
                    <th v-for="column in expandedTable.columns" :key="column">{{ columnLabel(column) }}</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="(row, rowIndex) in expandedTable.rows" :key="rowIndex">
                    <td v-for="column in expandedTable.columns" :key="column">{{ formatCell(row[column]) }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>
      </div>
      <MerchantAnalysisReport v-if="reportOpen" :report="analysisReport" @close="reportOpen = false" />
      <div v-if="toastMessage" class="app-toast">{{ toastMessage }}</div>
    </Teleport>
  </article>
</template>

<script setup>
import { computed, ref } from 'vue'
import { AlertTriangle, ArrowRight, BookOpenCheck, Check, CheckCircle2, Copy, Download, FileChartColumnIncreasing, Filter, Info, Maximize2, Sparkles, ThumbsDown, ThumbsUp, X } from 'lucide-vue-next'
import { buildAnalysisReport, hasAnalysisReportContent } from '../api/analysisReport'
import {
  collapseWhitespace,
  compactFixed,
  delimitedContents,
  humanizeIdentifier,
  isAsciiIdentifier,
  markdownLine,
  replaceAllLiteral,
  replaceCharacters,
  safeFileName as normalizedFileName,
  stripDatabaseQualifier
} from '../utils/textParsing'
import MerchantAnalysisReport from './MerchantAnalysisReport.vue'
import MetricLineChart from './MetricLineChart.vue'

const props = defineProps({
  id: String,
  anchorId: String,
  role: {
    type: String,
    required: true
  },
  text: {
    type: String,
    required: true
  },
  question: {
    type: String,
    default: ''
  },
  steps: {
    type: Array,
    default: () => []
  },
  tables: {
    type: Array,
    default: () => []
  },
  dataRows: {
    type: Array,
    default: () => []
  },
  dataSections: {
    type: Array,
    default: () => []
  },
  merchantExperience: {
    type: Object,
    default: () => ({})
  },
  clarification: {
    type: Object,
    default: null
  },
  feedbackStatus: {
    type: Object,
    default: () => ({})
  },
  workspaceMode: Boolean
})

const emit = defineEmits(['feedback', 'ask', 'confirm-clarification', 'metric-definition-action', 'knowledge-proposal-action'])

const displayTime = new Date().toLocaleString('zh-CN', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit'
})

const toastMessage = ref('')
const expandedTable = ref(null)
const reportOpen = ref(false)
const tableFilters = ref({})
let toastTimer = null

const experienceAlerts = computed(() => (props.merchantExperience?.anomalyAlerts || []).filter(Boolean).slice(0, 3))
const businessAdvice = computed(() => (props.merchantExperience?.businessAdvice || []).filter(Boolean).slice(0, 2))
const drillDownActions = computed(() => (props.merchantExperience?.drillDownActions || []).filter(action => action?.question).slice(0, 4))
const analysisReport = computed(() => buildAnalysisReport({
  question: props.question,
  answer: props.text,
  tables: props.tables,
  dataRows: props.dataRows,
  dataSections: props.dataSections,
  merchantExperience: props.merchantExperience
}))
const canBuildReport = computed(() => hasAnalysisReportContent(analysisReport.value))
const analysisScope = computed(() => props.merchantExperience?.analysisScope || {})
const analysisScopeVisible = computed(() => Boolean(
  analysisScope.value?.scopeDisclosureRequired || analysisScope.value?.mode === 'topic_workspace'
))
const analysisScopeText = computed(() => {
  const scope = analysisScope.value || {}
  const topics = (scope.topics || []).filter(Boolean)
  const label = topics.length ? topics.join(' + ') : '待确认'
  const modeLabels = {
    topic_workspace: '联合分析 Workspace',
    adaptive_workspace: '自动语义范围',
    open_discovery: '开放诊断范围',
    explicit_topic_scope: '用户限定范围'
  }
  const mode = modeLabels[scope.mode] || '自动语义范围'
  return `${mode}：${label}（置信度 ${Math.round(Number(scope.confidence || 0) * 100)}%）`
})
const knowledgeProposals = computed(() => (props.merchantExperience?.knowledgeSuggestions || [])
  .filter(item => item?.suggestionId)
  .slice(0, 3))
const disclosureItems = computed(() => {
  return (props.merchantExperience?.metricDisclosures || [])
    .map(item => item.description || item.displayName || item.metricKey)
    .filter(Boolean)
    .slice(0, 2)
})
const metricDefinitionCards = computed(() => {
  return (props.merchantExperience?.metricDisclosures || [])
    .filter(item => item?.description || item?.displayName || item?.metricKey)
    .slice(0, 2)
})
const traceabilityItems = computed(() => {
  const traceability = props.merchantExperience?.traceability || {}
  const items = []
  if (traceability.timeRange) items.push(`时间：${traceability.timeRange}`)
  if (traceability.dataUpdatedAt) items.push(`更新：${traceability.dataUpdatedAt}`)
  if (traceability.evidenceStatus) items.push(traceability.evidenceStatus === 'verified' ? '证据已校验' : '证据部分覆盖')
  if (traceability.sourceSummary) items.push(traceability.sourceSummary)
  return items.slice(0, 4)
})
const clarificationCard = computed(() => {
  const loop = props.merchantExperience?.humanLoop || {}
  const card = loop.confirmationCard || {}
  const direct = props.clarification || {}
  const question = card.question || direct.question || ''
  if (!question) return null
  return {
    type: card.type || direct.type || '',
    title: card.title || confirmationTitle(card.type || direct.type),
    question,
    stageLabel: direct.stage || loop.status || '需要确认',
    statusLabel: loop.status === 'resolved' ? '已确认' : '等待选择',
    checkpoint: loop.checkpoint || {}
  }
})
const clarificationOptions = computed(() => {
  const loopOptions = props.merchantExperience?.humanLoop?.confirmationCard?.options || []
  const directOptions = props.clarification?.options || []
  return [...loopOptions, ...directOptions]
    .map(item => String(item || '').trim())
    .filter((item, index, items) => item && items.indexOf(item) === index)
    .slice(0, 6)
})
const clarificationMetricPreview = computed(() => {
  const profile = props.merchantExperience?.merchantProfileSummary || {}
  const constraints = props.merchantExperience?.appliedMemoryConstraints || []
  const metrics = [
    ...(profile.preferredMetrics || []),
    ...constraints.flatMap(item => item?.targetMetrics || [])
  ]
  return metrics.map(item => String(item || '').trim()).filter(Boolean).slice(0, 5)
})
const clarificationTablePreview = computed(() => {
  const freshnessTables = props.merchantExperience?.dataFreshness?.tables || []
  const securityTables = props.merchantExperience?.securityAudit?.tables || []
  return [...freshnessTables, ...securityTables]
    .map(table => tableLabel(table))
    .filter((item, index, items) => item && items.indexOf(item) === index)
    .slice(0, 4)
})

function askFollowUp(question) {
  const text = String(question || '').trim()
  if (!text) return
  emit('ask', text)
}

function confirmClarification(option) {
  emit('confirm-clarification', {
    value: option,
    type: clarificationCard.value?.type || '',
    checkpoint: clarificationCard.value?.checkpoint || {},
    threadId: clarificationCard.value?.checkpoint?.threadId || ''
  })
}

function handleMetricDefinitionAction(item, action) {
  emit('metric-definition-action', {
    action,
    answerId: props.id || '',
    metricKey: item?.metricKey || '',
    displayName: item?.displayName || item?.metricKey || '',
    description: item?.description || '',
    formula: item?.formula || '',
    semanticRef: item?.semanticRef || item?.semanticRefId || '',
    sourceTable: item?.sourceTable || item?.table || '',
    note: action === 'question' ? '商家对当前披露口径提出疑问' : ''
  })
  showToast(action === 'question' ? '已提交口径疑问，等待平台审核' : '已记录为本商家的默认口径偏好')
}

function handleKnowledgeProposalAction(item, action, conflictResolution = '') {
  emit('knowledge-proposal-action', {
    action,
    conflictResolution,
    answerId: props.id || '',
    suggestionId: item?.suggestionId || '',
    scopeType: item?.scopeType || item?.scope || 'merchant'
  })
}

function conflictResolutionAllows(item, resolution) {
  return (item?.conflictCheck?.resolutionOptions || []).includes(resolution)
}

function knowledgeProposalAllows(item, action) {
  const configured = Array.isArray(item?.allowedActions) ? item.allowedActions : []
  if (configured.length) return configured.includes(action)
  if ((item?.scopeType || item?.scope) === 'platform') return ['suggest', 'skip'].includes(action)
  return ['accept', 'suggest', 'skip'].includes(action)
}

function knowledgeProposalActionable(item) {
  return ['candidate', 'review_required', 'pending', 'reviewed', ''].includes(String(item?.status || '').toLowerCase())
}

function knowledgeProposalStatusText(item) {
  const status = String(item?.status || '').toLowerCase()
  if (status === 'merchant_active') return '已保存为本店设置，后续对话会按此执行'
  if (status === 'platform_suggested') return '已提交给得物平台，审核通过后更新公共口径'
  if (status === 'dismissed') return '已忽略，本条内容不会保存'
  if (item?.actionPending) return '正在处理…'
  if ((item?.scopeType || item?.scope) === 'platform') return '涉及平台口径，只能提交给平台审核'
  return '可保存为本店设置，也可以反馈给得物平台'
}

function confirmationTitle(type) {
  const mapping = {
    time_window: '确认分析时间范围',
    metric_focus: '确认指标口径',
    priority_goal: '确认优化目标',
    skill_confirm: '是否开始深度分析',
    business_scope: '确认业务范围'
  }
  return mapping[type] || '确认分析口径'
}

const visibleDataSections = computed(() => {
  const sections = props.dataSections || []
  if (sections.length <= 1) {
    return sections
  }
  const answerTitles = answerSelectedTitles(props.text)
  if (!answerTitles.size) {
    return sections
  }
  const selected = sections.filter(section => answerTitles.has(normalizeSectionTitle(section?.title || '')))
  return selected.length ? selected : sections
})

const metricSummarySections = computed(() => {
  const structuredSections = visibleDataSections.value
    .map((section) => {
      const rows = extractDisplayRows(section?.dataRows || [])
      return metricSummaryFromRows(rows, section?.title || '', section?.dorisTables || [])
    })
    .filter(Boolean)
  if (structuredSections.length) {
    return structuredSections
  }
  return [metricSummaryFromRows(extractDisplayRows(props.dataRows || []), '', props.tables || [])].filter(Boolean)
})

const chartSections = computed(() => {
  const structuredSections = visibleDataSections.value
    .map((section) => {
      const rows = section?.dataRows || []
      return {
        title: section?.title || '',
        tables: section?.dorisTables || [],
        rows
      }
    })
    .filter(section => isMetricSeriesRows(section.rows))
  if (structuredSections.length) {
    return structuredSections
  }
  if (isMetricSeriesRows(props.dataRows || [])) {
    return [{
      title: '',
      tables: props.tables || [],
      rows: props.dataRows || []
    }]
  }
  return []
})

const aggregateSections = computed(() => {
  const structuredSections = visibleDataSections.value
    .map((section) => {
      const rows = extractDisplayRows(section?.dataRows || [])
      return {
        title: section?.title || '',
        tables: section?.dorisTables || [],
        rows,
        mode: inferAggregateMode(section)
      }
    })
    .filter(section => isAggregateRows(section.rows))
  if (structuredSections.length) {
    return structuredSections
  }
  const rows = extractDisplayRows(props.dataRows || [])
  if (!isAggregateRows(rows)) {
    return []
  }
  return [{
    title: '',
    tables: props.tables || [],
    rows,
    mode: inferAggregateMode({})
  }]
})

const tableSections = computed(() => {
  const structuredSections = visibleDataSections.value
    .map((section) => {
      const rows = extractDisplayRows(section?.dataRows || [])
      return {
        title: section?.title || '',
        tables: section?.dorisTables || [],
        rows,
        columns: collectColumns(rows)
      }
    })
    .filter(section => section.rows.length && !isMetricSeriesRows(section.rows) && !isAggregateRows(section.rows) && !isSingleMetricRows(section.rows))
  if (structuredSections.length) {
    return structuredSections
  }
  if ((props.dataSections || []).length) {
    return []
  }
  const rows = extractDisplayRows(props.dataRows || [])
  if (!rows.length) {
    return []
  }
  if ((props.dataRows || []).length && (props.dataRows || []).every(row => Object.prototype.hasOwnProperty.call(row || {}, '__metricKey'))) {
    return []
  }
  if (isMetricSeriesRows(rows) || isAggregateRows(rows) || isSingleMetricRows(rows)) {
    return []
  }
  const hasMetricRows = (props.dataRows || []).some(row => Object.prototype.hasOwnProperty.call(row || {}, 'metric_name'))
  const hasDetailRows = (props.dataRows || []).some(row => !Object.prototype.hasOwnProperty.call(row || {}, 'metric_name'))
  return [{
    title: hasMetricRows && hasDetailRows ? '明细表' : '',
    tables: props.tables || [],
    rows,
    columns: collectColumns(rows)
  }]
})

const answerBlocks = computed(() => {
  const sections = []
  let current = { title: '', text: '', items: [] }
  const flush = () => {
    if (current.title || current.text || current.items.length) {
      sections.push(current)
      current = { title: '', text: '', items: [] }
    }
  }

  for (const rawLine of String(props.text || '').split('\n')) {
    const parsed = markdownLine(rawLine)
    if (parsed.kind === 'empty') {
      flush()
      continue
    }
    if (parsed.kind === 'heading') {
      flush()
      current.title = parsed.text
      continue
    }
    if (parsed.kind === 'bullet') {
      current.items.push(parsed.text)
      continue
    }
    if (!current.text) {
      current.text = parsed.text
    } else {
      current.text += `\n${parsed.text}`
    }
  }
  flush()
  return sections.length ? sections : [{ title: '', text: props.text, items: [] }]
})

function answerSelectedTitles(text) {
  return new Set(delimitedContents(text, '【', '】').map(normalizeSectionTitle))
}

function normalizeSectionTitle(title) {
  return collapseWhitespace(replaceCharacters(title, '／/', '-'), '').trim()
}

function extractDisplayRows(rows) {
  if (!rows.length) {
    return []
  }
  const detailRows = rows.filter(row => !Object.prototype.hasOwnProperty.call(row || {}, 'metric_name'))
  return detailRows.length ? detailRows : rows
}

function isMetricSeriesRows(rows) {
  return Array.isArray(rows)
    && rows.length > 0
    && rows.every(row =>
      Object.prototype.hasOwnProperty.call(row || {}, 'metric_name')
      && Object.prototype.hasOwnProperty.call(row || {}, 'pt')
      && Object.prototype.hasOwnProperty.call(row || {}, 'value'))
}

function isAggregateRows(rows) {
  return Array.isArray(rows)
    && rows.length > 0
    && rows.every(row =>
      Object.prototype.hasOwnProperty.call(row || {}, 'group_value')
      && Object.prototype.hasOwnProperty.call(row || {}, 'metric_value'))
}

function isSingleMetricRows(rows) {
  if (!Array.isArray(rows) || rows.length !== 1) return false
  return metricValueColumns(rows[0]).length === 1
}

function metricSummaryFromRows(rows, title, tables) {
  if (!isSingleMetricRows(rows)) {
    return null
  }
  const row = rows[0]
  const valueColumn = metricValueColumns(row)[0]
  return {
    label: title && !looksLikeRawField(title) ? presentSectionTitle(title) : columnLabel(valueColumn),
    value: formatMetricSummaryValue(row[valueColumn], valueColumn),
    valueColumn,
    tables
  }
}

function metricValueColumns(row) {
  return Object.keys(row || {}).filter((column) => {
    if (isIdentifierColumn(column) || column.startsWith('__')) return false
    return numericValue(row[column]) !== null
  })
}

function isIdentifierColumn(column) {
  const text = String(column || '').toLowerCase()
  return text === 'pt'
    || text === 'seller_id'
    || text === 'merchant_id'
    || text.endsWith('_id')
    || text.endsWith('_no')
}

function numericValue(value) {
  if (value === null || value === undefined || value === '' || typeof value === 'boolean') return null
  const numeric = Number(replaceAllLiteral(String(value), ','))
  return Number.isFinite(numeric) ? numeric : null
}

function formatMetricSummaryValue(value) {
  const numeric = numericValue(value)
  if (numeric === null) return formatCell(value)
  return formatCompactNumber(numeric)
}

function formatCompactNumber(value) {
  if (Math.abs(value) >= 10000) return `${compactFixed(value / 10000)}万`
  if (Number.isInteger(value)) return String(value)
  return compactFixed(value)
}

function collectColumns(rows) {
  const columns = []
  for (const row of rows) {
    for (const column of Object.keys(row || {})) {
      if (column.startsWith('__')) continue
      if (!columns.includes(column)) {
        columns.push(column)
      }
    }
  }
  return columns
}

const COLUMN_LABEL_LIMIT = 10

const metricPresentation = computed(() => {
  const result = {}
  for (const item of props.merchantExperience?.metricDisclosures || []) {
    const key = String(item?.metricKey || '').trim().toLowerCase()
    if (key) result[key] = item
  }
  return result
})

function tableLabel(table) {
  return stripDatabaseQualifier(table) || '相关数据'
}

function columnLabel(column) {
  const rawColumn = String(column || '').trim()
  const governed = metricPresentation.value[rawColumn.toLowerCase()]?.displayName
  const label = governed || (isAsciiIdentifier(rawColumn) ? humanizeIdentifier(rawColumn) : rawColumn)
  return shortenLabel(label || rawColumn)
}

function shortenLabel(label) {
  const value = String(label || '').trim()
  return value.length <= COLUMN_LABEL_LIMIT ? value : value.slice(0, COLUMN_LABEL_LIMIT)
}
function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function inferAggregateMode(section) {
  return String(section?.resultRole || '').trim().toUpperCase() === 'TOPN' ? 'topn' : 'group'
}

function presentSectionTitle(title) {
  if (!title) {
    return '明细结果'
  }
  const normalized = String(title)
    .split('-')
    .filter(Boolean)
    .join(' / ')
  return looksLikeRawField(normalized) ? columnLabel(normalized) : normalized
}

function looksLikeRawField(title) {
  return isAsciiIdentifier(String(title || ''))
}

function resolveGroupLabel() {
  return '分组对象'
}

function resolveMetricLabel() {
  return '指标值'
}

function formatAggregateGroup(row) {
  return formatCell(row?.group_value)
}

function formatAggregateMetric(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return formatCell(value)
  }
  return Number.isInteger(numeric) ? `${numeric}` : compactFixed(numeric)
}

function formatAggregateCount(value) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return formatCell(value)
  }
  return `${numeric}条记录`
}

function tableFilterOpen(index) {
  return Boolean(tableFilters.value[index]?.open)
}

function tableFilterQuery(index) {
  return tableFilters.value[index]?.query || ''
}

function toggleTableFilter(index) {
  const current = tableFilters.value[index] || { open: false, query: '' }
  const nextOpen = !current.open
  tableFilters.value = {
    ...tableFilters.value,
    [index]: {
      ...current,
      open: nextOpen
    }
  }
  showToast(nextOpen ? '已打开筛选' : '已关闭筛选')
}

function setTableFilterQuery(index, query) {
  const current = tableFilters.value[index] || { open: true, query: '' }
  tableFilters.value = {
    ...tableFilters.value,
    [index]: {
      ...current,
      open: true,
      query
    }
  }
}

function filteredTableRows(section, index) {
  const rows = section?.rows || []
  const query = tableFilterQuery(index).trim().toLowerCase()
  if (!query) {
    return rows
  }
  const columns = section?.columns || []
  return rows.filter(row => columns.some(column => {
    const label = columnLabel(column).toLowerCase()
    const value = formatCell(row?.[column]).toLowerCase()
    return label.includes(query) || value.includes(query)
  }))
}

async function copyTable(section, index) {
  const rows = filteredTableRows(section, index)
  const text = tableToDelimitedText(section, rows, '\t')
  const copied = await writeClipboardText(text)
  showToast(copied ? '已复制表格' : '复制失败，请重试')
}

function downloadTable(section, index) {
  const rows = filteredTableRows(section, index)
  const csv = `\ufeff${tableToDelimitedText(section, rows, ',')}`
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${safeFileName(presentSectionTitle(section?.title) || '查询结果')}.csv`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
  showToast('已下载 CSV')
}

function openExpandedTable(section, index) {
  expandedTable.value = {
    title: presentSectionTitle(section?.title),
    columns: section?.columns || [],
    rows: filteredTableRows(section, index)
  }
  showToast('已打开放大视图')
}

function closeExpandedTable() {
  expandedTable.value = null
}

function tableToDelimitedText(section, rows, delimiter) {
  const columns = section?.columns || []
  const escapeValue = delimiter === ',' ? csvEscape : textEscape
  const header = columns.map(column => escapeValue(columnLabel(column))).join(delimiter)
  const body = (rows || []).map(row =>
    columns.map(column => escapeValue(formatCell(row?.[column]))).join(delimiter)
  )
  return [header, ...body].join('\n')
}

function csvEscape(value) {
  const text = String(value ?? '')
  if (text.includes('"') || text.includes(',') || text.includes('\n')) {
    return `"${replaceAllLiteral(text, '"', '""')}"`
  }
  return text
}

function textEscape(value) {
  return replaceCharacters(String(value ?? ''), '\t\n', ' ')
}

function safeFileName(name) {
  return normalizedFileName(name, '查询结果', 60)
}

async function writeClipboardText(text) {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Fall through to the textarea fallback.
  }
  try {
    const textarea = document.createElement('textarea')
    textarea.value = text
    textarea.setAttribute('readonly', '')
    textarea.style.position = 'fixed'
    textarea.style.opacity = '0'
    document.body.appendChild(textarea)
    textarea.select()
    const ok = document.execCommand('copy')
    textarea.remove()
    return ok
  } catch {
    return false
  }
}

function showToast(message) {
  toastMessage.value = message
  if (toastTimer) {
    window.clearTimeout(toastTimer)
  }
  toastTimer = window.setTimeout(() => {
    toastMessage.value = ''
  }, 1800)
}

</script>
