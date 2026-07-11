<template>
  <section v-if="chart" class="rail-chart-card">
    <div class="rail-chart-head"><div><span>{{ chart.kindLabel }}</span><h3>{{ chart.title }}</h3></div><b>{{ chart.rows.length }} 项</b></div>
    <MetricLineChart v-if="chart.kind === 'line'" :title="chart.title" :rows="chart.rows" :tables="chart.tables" />
    <div v-else class="mini-bars">
      <div v-for="row in chart.rows.slice(0, 7)" :key="row.label">
        <p><span>{{ row.label }}</span><b>{{ format(row.value) }}</b></p>
        <i><em :style="{ width: `${Math.max(6, row.value / chart.max * 100)}%` }" /></i>
      </div>
    </div>
    <div class="chart-analysis"><Sparkles :size="15" /><p>{{ insight }}</p></div>
  </section>
  <section v-else class="rail-empty-state"><ChartNoAxesCombined :size="26" /><b>暂无指标图表</b><p>询问 GMV、订单量、退款率等指标后，图表会在这里随回答更新。</p></section>
</template>

<script setup>
import { computed } from 'vue'
import { ChartNoAxesCombined, Sparkles } from 'lucide-vue-next'
import MetricLineChart from './MetricLineChart.vue'
const props = defineProps({ message: { type: Object, default: null } })
const chart = computed(() => {
  const message = props.message || {}
  const sections = message.dataSections?.length ? message.dataSections : [{ title: '', dataRows: message.dataRows || [], dorisTables: message.tables || [] }]
  for (const section of sections) {
    const raw = section.dataRows || []
    const timeRows = raw.map(row => ({ pt: row.pt || row.date || row.dt, value: Number(row.value ?? row.metric_value ?? row.cnt) })).filter(row => row.pt && Number.isFinite(row.value))
    if (timeRows.length > 1) return { kind: 'line', kindLabel: '按日期趋势', title: section.title || '核心指标趋势', rows: timeRows, tables: section.dorisTables || [], max: 1 }
    const dimensionRows = raw.map(row => ({ label: String(row.group_value ?? row.name ?? row.category ?? ''), value: Number(row.metric_value ?? row.value ?? row.cnt) })).filter(row => row.label && Number.isFinite(row.value))
    if (dimensionRows.length > 1) return { kind: 'bar', kindLabel: '维度对比', title: section.title || '指标构成', rows: dimensionRows, tables: section.dorisTables || [], max: Math.max(...dimensionRows.map(row => row.value), 1) }
  }
  return null
})
const insight = computed(() => {
  if (!chart.value) return ''
  const values = chart.value.rows.map(row => Number(row.value))
  const first = values[0] || 0
  const last = values.at(-1) || 0
  if (chart.value.kind === 'line') return last > first ? `指标较期初上升 ${format(last - first)}，建议继续结合异常日期定位驱动因素。` : `指标较期初下降 ${format(first - last)}，建议优先排查降幅最大的日期。`
  const top = chart.value.rows.reduce((a, b) => a.value > b.value ? a : b)
  return `${top.label} 当前占比/数值最高，是本轮分析最值得优先关注的维度。`
})
function format(value) { return new Intl.NumberFormat('zh-CN', { maximumFractionDigits: 2 }).format(value || 0) }
</script>
