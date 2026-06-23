<template>
  <section class="metric-chart-card">
    <div class="metric-chart-head">
      <div>
        <p class="metric-chart-kicker">{{ kicker }}</p>
        <h3>{{ displayTitle }}</h3>
      </div>
      <div class="metric-chart-stat">
        <span>{{ summaryLabel }}</span>
        <strong>{{ summaryValue }}</strong>
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

    <div v-if="tables?.length" class="table-tags section-table-tags">
      <span v-for="table in tables" :key="table">{{ tableLabel(table) }}</span>
    </div>
  </section>
</template>

<script setup>
import { computed } from 'vue'

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

const tableLabels = {
  ads_merchant_profile: '店铺经营指标',
  dwm_trade_order_detail_di: '订单数据',
  dwm_trade_refund_detail_di: '退款/售后数据',
  dwm_goods_detail_df: '商品数据',
  dwm_cs_ticket_detail_di: '客服工单数据',
  dwm_cs_repay_detail_df: '赔付数据',
  dwm_coupon_detail_di: '优惠券数据',
  dwm_scm_detail_di: '供应链履约数据',
  dim_merchant_df: '商家资料',
  dwd_merchant_appeal_detail_df: '申诉数据',
  dwd_merchant_deposit_recharge_df: '保证金数据'
}

function tableLabel(table) {
  const normalized = String(table || '').replace(/^yshopping\./, '').trim()
  return tableLabels[normalized] || '相关业务数据'
}

const chartWidth = 640
const chartHeight = 260
const padding = { top: 20, right: 18, bottom: 44, left: 52 }

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
  return titleLooksLikeAmount(displayTitle.value) ? '金额趋势' : '指标趋势'
})

const summaryLabel = computed(() => (titleLooksLikeAmount(displayTitle.value) ? '总计' : '峰值'))

const summaryValue = computed(() => {
  if (!normalizedRows.value.length) return '-'
  if (titleLooksLikeAmount(displayTitle.value)) {
    const total = normalizedRows.value.reduce((sum, row) => sum + row.value, 0)
    return formatMetricValue(total)
  }
  return formatMetricValue(peakPoint.value?.value ?? 0)
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
  return Number(value).toFixed(2).replace(/\.00$/, '')
}

function compactDate(text) {
  const value = String(text || '')
  const match = value.match(/(\d{4})-(\d{2})-(\d{2})/)
  if (!match) return value
  return `${match[2]}-${match[3]}`
}

function titleLooksLikeAmount(title) {
  return /金额|gmv|销售额|成交额|流水/i.test(String(title || ''))
}
</script>
