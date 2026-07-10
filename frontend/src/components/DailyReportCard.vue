<template>
  <section :class="['daily-card', { compact }]">
    <div class="daily-head">
      <div>
        <p class="muted">每日 10 点推送</p>
        <h2>下午好，这是昨日的经营数据</h2>
      </div>
      <button
        v-if="compact"
        type="button"
        class="daily-toggle"
        @click="expanded = !expanded"
      >
        {{ expanded ? '收起' : '查看' }}
        <ChevronDown :class="{ open: expanded }" :size="16" />
      </button>
      <div v-else class="date-pill">
        <b>{{ day }}</b>
        <span>{{ month }}</span>
      </div>
    </div>
    <p class="daily-summary">{{ summary }}</p>
    <div v-if="alertItems.length" class="daily-alerts">
      <div v-for="alert in alertItems" :key="alert.message" class="daily-alert">
        <AlertTriangle :size="15" />
        <span>{{ alert.message }}</span>
      </div>
    </div>
    <div v-if="!compact || expanded" class="metric-grid">
      <div v-for="item in metricItems" :key="item.label" class="metric-item">
        <span>{{ item.label }}</span>
        <strong>{{ item.value }}</strong>
      </div>
    </div>
    <div v-if="(!compact || expanded) && drillActions.length" class="daily-drill-panel">
      <h3>快捷下钻</h3>
      <button
        v-for="action in drillActions"
        :key="action.label"
        type="button"
        @click="askFollowUp(action.question)"
      >
        <span>{{ action.label }}</span>
        <ArrowRight :size="14" />
      </button>
    </div>
    <div v-if="!compact || expanded" class="advice-panel">
      <h3>经营建议</h3>
      <div v-for="(suggestion, index) in report.suggestions" :key="suggestion" class="advice-row">
        <p>{{ suggestion }}</p>
        <button
          type="button"
          :class="{ adopted: adoptedSuggestions.has(index) }"
          @click="adopt(index)"
        >
          {{ adoptedSuggestions.has(index) ? '已采纳' : '采纳' }}
        </button>
      </div>
    </div>
    <p v-if="(!compact || expanded) && sourceText" class="daily-source">{{ sourceText }}</p>
  </section>
</template>

<script setup>
import { computed, ref } from 'vue'
import { AlertTriangle, ArrowRight, ChevronDown } from 'lucide-vue-next'

const props = defineProps({
  report: {
    type: Object,
    required: true
  },
  compact: {
    type: Boolean,
    default: false
  }
})
const emit = defineEmits(['ask'])

const adoptedSuggestions = ref(new Set())
const expanded = ref(false)

const metricItems = computed(() => {
  const metrics = props.report.metrics || {}
  return Object.entries(metrics).map(([label, value]) => ({
    label,
    value: formatValue(label, value)
  }))
})
const alertItems = computed(() => (props.report.anomalyAlerts || props.report.anomaly_alerts || []).slice(0, 2))
const drillActions = computed(() => (props.report.drillDownActions || props.report.drill_down_actions || []).slice(0, 3))

const date = computed(() => props.report.date || '2026-05-23')
const day = computed(() => new Date(date.value).getDate())
const month = computed(() => `${new Date(date.value).getMonth() + 1}月`)
const summary = computed(() => {
  const gmv = Number((props.report.metrics || {})['昨日总gmv金额'] || 0)
  if (alertItems.value.length) return '昨日经营有需要关注的变化，建议优先处理下方提醒。'
  return gmv > 0 ? '昨日已有成交数据，建议继续关注转化和售后表现。' : '暂无昨日经营数据，可以先关注商品、客服和履约基础配置。'
})
const sourceText = computed(() => {
  const trace = props.report.traceability || {}
  if (!trace.sourceSummary && !trace.timeRange) return ''
  return [trace.sourceSummary, trace.timeRange ? `范围：${trace.timeRange}` : ''].filter(Boolean).join(' · ')
})

function formatValue(label, value) {
  const numeric = Number(value || 0)
  if (label.includes('金额') || label.toLowerCase().includes('gmv')) {
    return `￥${numeric.toLocaleString('zh-CN', { maximumFractionDigits: 2 })}`
  }
  return numeric.toLocaleString('zh-CN')
}

function adopt(index) {
  const next = new Set(adoptedSuggestions.value)
  next.add(index)
  adoptedSuggestions.value = next
}

function askFollowUp(question) {
  if (!question) return
  emit('ask', question)
}
</script>
