<template>
  <section class="metric-chart-card">
    <div class="metric-chart-head">
      <div>
        <p class="metric-chart-kicker">{{ kicker }}</p>
        <h3>{{ displayTitle }}</h3>
      </div>
      <div class="metric-chart-actions" aria-label="图表操作">
        <button type="button" title="放大查看" @click="expanded = true">
          <Maximize2 :size="14" />
        </button>
        <button type="button" title="复制图表数据" @click="copyChartData">
          <Copy :size="14" />
        </button>
      </div>
    </div>

    <svg
      class="metric-chart-svg"
      viewBox="0 0 640 260"
      role="img"
      :aria-label="`${displayTitle} 趋势图`"
      preserveAspectRatio="none"
    >
      <g v-for="tick in yTicks" :key="`grid-${tick.value}`">
        <line
          :x1="padding.left"
          :x2="chartWidth - padding.right"
          :y1="tick.y"
          :y2="tick.y"
          class="metric-chart-grid"
        />
        <text
          :x="padding.left - 12"
          :y="tick.y + 4"
          text-anchor="end"
          class="metric-chart-y-label"
        >
          {{ formatAxisValue(tick.value) }}
        </text>
      </g>

      <g v-for="label in xLabels" :key="`label-${label.index}`">
        <line
          :x1="label.x"
          :x2="label.x"
          :y1="padding.top"
          :y2="chartHeight - padding.bottom"
          class="metric-chart-column"
        />
        <text
          :x="label.x"
          :y="chartHeight - padding.bottom + 24"
          text-anchor="middle"
          class="metric-chart-x-label"
        >
          {{ label.text }}
        </text>
      </g>

      <path :d="areaPath" class="metric-chart-area" />
      <path :d="linePath" class="metric-chart-line" />

      <g v-for="point in points" :key="`${point.label}-${point.index}`">
        <circle :cx="point.x" :cy="point.y" r="4.5" class="metric-chart-point-shadow" />
        <circle :cx="point.x" :cy="point.y" r="3.2" class="metric-chart-point" />
      </g>
    </svg>

    <div class="metric-chart-footer">
      <span>{{ rangeLabel }}</span>
      <span>{{ pointCount }} 个采样点</span>
      <span>峰值 {{ peakLabel }}</span>
    </div>

  </section>

  <Teleport to="body">
    <div v-if="expanded" class="result-modal-backdrop" @click.self="expanded = false">
      <section class="result-modal result-chart-modal" role="dialog" aria-modal="true" :aria-label="`${displayTitle} 放大查看`">
        <div class="result-modal-head">
          <div>
            <p>{{ rangeLabel }} · {{ pointCount }} 个采样点</p>
            <h3>{{ displayTitle }}</h3>
          </div>
          <button type="button" title="关闭" @click="expanded = false">
            <X :size="18" />
          </button>
        </div>
        <svg
          class="metric-chart-svg metric-chart-modal-svg"
          viewBox="0 0 640 260"
          role="img"
          :aria-label="`${displayTitle} 趋势图`"
          preserveAspectRatio="none"
        >
          <g v-for="tick in yTicks" :key="`modal-grid-${tick.value}`">
            <line
              :x1="padding.left"
              :x2="chartWidth - padding.right"
              :y1="tick.y"
              :y2="tick.y"
              class="metric-chart-grid"
            />
            <text
              :x="padding.left - 12"
              :y="tick.y + 4"
              text-anchor="end"
              class="metric-chart-y-label"
            >
              {{ formatAxisValue(tick.value) }}
            </text>
          </g>

          <g v-for="label in xLabels" :key="`modal-label-${label.index}`">
            <line
              :x1="label.x"
              :x2="label.x"
              :y1="padding.top"
              :y2="chartHeight - padding.bottom"
              class="metric-chart-column"
            />
            <text
              :x="label.x"
              :y="chartHeight - padding.bottom + 24"
              text-anchor="middle"
              class="metric-chart-x-label"
            >
              {{ label.text }}
            </text>
          </g>

          <path :d="areaPath" class="metric-chart-area" />
          <path :d="linePath" class="metric-chart-line" />

          <g v-for="point in points" :key="`modal-point-${point.label}-${point.index}`">
            <circle :cx="point.x" :cy="point.y" r="4.5" class="metric-chart-point-shadow" />
            <circle :cx="point.x" :cy="point.y" r="3.2" class="metric-chart-point" />
          </g>
        </svg>
        <div class="metric-chart-footer result-modal-footer">
          <span>{{ rangeLabel }}</span>
          <span>{{ pointCount }} 个采样点</span>
          <span>峰值 {{ peakLabel }}</span>
        </div>
      </section>
    </div>
    <div v-if="toastMessage" class="app-toast">{{ toastMessage }}</div>
  </Teleport>
</template>

<script setup>
import { computed, ref } from 'vue'
import { Copy, Maximize2, X } from 'lucide-vue-next'
import { compactFixed, isoDateParts } from '../utils/textParsing'

const props = defineProps({
  title: {
    type: String,
    required: true
  },
  rows: {
    type: Array,
    default: () => []
  },
  tables: {
    type: Array,
    default: () => []
  }
})

const chartWidth = 640
const chartHeight = 260
const padding = { top: 20, right: 18, bottom: 44, left: 52 }
const expanded = ref(false)
const toastMessage = ref('')
let toastTimer = null

const displayTitle = computed(() => props.title || '指标趋势')

const normalizedRows = computed(() => {
  return (props.rows || [])
    .map((row, index) => ({
      index,
      label: String(row?.pt || row?.date || row?.dt || ''),
      value: Number(row?.value ?? row?.cnt ?? 0)
    }))
    .filter(row => row.label)
})

const pointCount = computed(() => normalizedRows.value.length)

const metricValues = computed(() => normalizedRows.value.map(row => row.value))

const valueRange = computed(() => {
  if (!metricValues.value.length) {
    return { min: 0, max: 1, paddedMin: 0, paddedMax: 1 }
  }
  const min = Math.min(...metricValues.value)
  const max = Math.max(...metricValues.value)
  if (min === max) {
    const paddingValue = Math.max(1, Math.abs(max) * 0.08 || 1)
    return {
      min,
      max,
      paddedMin: min - paddingValue,
      paddedMax: max + paddingValue
    }
  }
  const paddingValue = (max - min) * 0.14
  return {
    min,
    max,
    paddedMin: Math.max(0, min - paddingValue),
    paddedMax: max + paddingValue
  }
})

const points = computed(() => {
  const rows = normalizedRows.value
  if (!rows.length) return []
  const usableWidth = chartWidth - padding.left - padding.right
  const usableHeight = chartHeight - padding.top - padding.bottom
  const denominator = Math.max(1, rows.length - 1)
  const valueSpan = valueRange.value.paddedMax - valueRange.value.paddedMin || 1

  return rows.map((row, index) => {
    const x = padding.left + (usableWidth * index) / denominator
    const ratio = (row.value - valueRange.value.paddedMin) / valueSpan
    const y = chartHeight - padding.bottom - ratio * usableHeight
    return {
      ...row,
      x,
      y
    }
  })
})

const linePath = computed(() => {
  if (!points.value.length) return ''
  return points.value
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ')
})

const areaPath = computed(() => {
  if (!points.value.length) return ''
  const baseline = chartHeight - padding.bottom
  const line = points.value
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
    .join(' ')
  const last = points.value[points.value.length - 1]
  const first = points.value[0]
  return `${line} L ${last.x.toFixed(2)} ${baseline} L ${first.x.toFixed(2)} ${baseline} Z`
})

const yTicks = computed(() => {
  const tickCount = 4
  const ticks = []
  const span = valueRange.value.paddedMax - valueRange.value.paddedMin || 1
  const usableHeight = chartHeight - padding.top - padding.bottom

  for (let i = 0; i < tickCount; i += 1) {
    const ratio = i / (tickCount - 1)
    const value = valueRange.value.paddedMax - span * ratio
    const y = padding.top + usableHeight * ratio
    ticks.push({ value, y })
  }
  return ticks
})

const xLabels = computed(() => {
  const rows = points.value
  if (!rows.length) return []
  const maxLabels = Math.min(6, rows.length)
  const step = Math.max(1, Math.ceil(rows.length / maxLabels))
  return rows
    .filter((_, index) => index % step === 0 || index === rows.length - 1)
    .map(row => ({
      index: row.index,
      x: row.x,
      text: compactDate(row.label)
    }))
})

const peakPoint = computed(() => {
  if (!normalizedRows.value.length) {
    return null
  }
  return normalizedRows.value.reduce((best, row) => (row.value > best.value ? row : best), normalizedRows.value[0])
})

const kicker = computed(() => {
  if (!normalizedRows.value.length) return '趋势分析'
  return '指标趋势'
})

const peakLabel = computed(() => {
  if (!peakPoint.value) return '-'
  return `${compactDate(peakPoint.value.label)} / ${formatMetricValue(peakPoint.value.value)}`
})

const rangeLabel = computed(() => {
  if (!normalizedRows.value.length) return '-'
  const first = normalizedRows.value[0]
  const last = normalizedRows.value[normalizedRows.value.length - 1]
  return `${compactDate(first.label)} - ${compactDate(last.label)}`
})

function formatAxisValue(value) {
  const abs = Math.abs(value)
  if (abs >= 10000) return `${(value / 10000).toFixed(1)}w`
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`
  return Number(value).toFixed(abs < 10 ? 1 : 0)
}

function formatMetricValue(value) {
  if (!Number.isFinite(value)) return '-'
  if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(2)}万`
  if (Math.abs(value) >= 1000) return Number(value).toFixed(0)
  if (Math.abs(value) >= 100) return Number(value).toFixed(1)
  return compactFixed(value)
}

function compactDate(text) {
  const value = String(text || '')
  const parts = isoDateParts(value)
  return parts.length ? `${parts[1]}-${parts[2]}` : value
}

async function copyChartData() {
  const rows = normalizedRows.value.map(row => `${row.label}\t${formatMetricValue(row.value)}`)
  const text = [`${displayTitle.value}`, '日期\t数值', ...rows].join('\n')
  const copied = await writeClipboardText(text)
  showToast(copied ? '已复制趋势数据' : '复制失败，请重试')
}

async function writeClipboardText(text) {
  try {
    if (navigator?.clipboard?.writeText) {
      await navigator.clipboard.writeText(text)
      return true
    }
  } catch {
    // Use the textarea fallback below.
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
