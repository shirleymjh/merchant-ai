<template>
  <div class="report-backdrop" @click.self="$emit('close')">
    <section class="report-window" role="dialog" aria-modal="true" :aria-label="report.title">
      <header class="report-toolbar">
        <div><FileChartColumnIncreasing :size="20" /><span>经营分析报告</span></div>
        <nav>
          <button type="button" class="report-download" @click="download"><Download :size="16" />下载 HTML</button>
          <button type="button" title="关闭报告" @click="$emit('close')"><X :size="19" /></button>
        </nav>
      </header>
      <div class="report-scroll">
        <main class="report-page">
          <header class="report-hero">
            <span class="report-brand">YSHOPPING · MERCHANT INTELLIGENCE</span>
            <h1>{{ report.title }}</h1>
            <p>{{ report.summary || '基于本轮经营数据生成的结构化分析报告。' }}</p>
            <div class="report-meta">
              <span><CalendarRange :size="14" />{{ report.timeRange }}</span>
              <span><Clock3 :size="14" />{{ report.generatedAt }}</span>
              <span v-if="report.dataUpdatedAt"><DatabaseZap :size="14" />数据更新 {{ report.dataUpdatedAt }}</span>
            </div>
          </header>

          <section v-if="report.metricCards.length" class="report-metrics">
            <article v-for="card in report.metricCards" :key="card.label">
              <span>{{ card.label }}</span><strong>{{ card.value }}</strong><small>{{ card.context || report.timeRange }}</small>
            </article>
          </section>

          <article v-for="trend in report.trends" :key="trend.title" class="report-panel trend-panel">
            <div class="report-panel-title"><span>趋势表现</span><h2>{{ trend.title }}</h2></div>
            <div class="report-bars">
              <div v-for="point in trend.points" :key="`${point.label}-${point.value}`" class="report-bar-item">
                <div class="report-bar-track"><i :style="{ height: `${point.height}%` }" /></div>
                <b>{{ point.value }}</b><span>{{ point.label }}</span>
              </div>
            </div>
          </article>

          <section v-if="report.anomalies.length || report.actions.length" class="report-two-column">
            <article v-if="report.anomalies.length" class="report-panel risk-panel">
              <div class="report-panel-title"><span>经营解读</span><h2>异常与风险</h2></div>
              <div class="report-list">
                <div v-for="item in report.anomalies" :key="item.message"><b>{{ item.metric }}</b><p>{{ item.message }}</p></div>
              </div>
            </article>
            <article v-if="report.actions.length" class="report-panel action-panel">
              <div class="report-panel-title"><span>经营解读</span><h2>建议行动</h2></div>
              <div class="report-list">
                <div v-for="(item, index) in report.actions" :key="item"><b>{{ index + 1 }}</b><p>{{ item }}</p></div>
              </div>
            </article>
          </section>

          <article v-for="table in report.detailTables" :key="table.title" class="report-panel report-table-panel">
            <div class="report-panel-title"><span>数据明细</span><h2>{{ table.title }}</h2></div>
            <div class="report-table-scroll"><table><thead><tr><th v-for="column in table.columns" :key="column.key">{{ column.label }}</th></tr></thead>
              <tbody><tr v-for="(row, rowIndex) in table.rows" :key="rowIndex"><td v-for="column in table.columns" :key="column.key">{{ formatCell(row[column.key]) }}</td></tr></tbody>
            </table></div>
          </article>

          <article v-if="report.definitions.length" class="report-panel definition-panel">
            <div class="report-panel-title"><span>可信说明</span><h2>指标口径</h2></div>
            <div class="report-definitions"><div v-for="item in report.definitions" :key="item.name"><b>{{ item.name }}</b><p>{{ item.description }}</p></div></div>
          </article>

          <footer class="report-foot">
            <b>数据说明</b>
            <p>数据来源：{{ report.sources.join('、') || '本轮已查询的经营数据' }}。本报告直接使用当前会话已有结果生成，未为展示重复查询数据。</p>
            <span>内容为 AI 生成，仅供经营决策参考</span>
          </footer>
        </main>
      </div>
    </section>
  </div>
</template>

<script setup>
import { CalendarRange, Clock3, DatabaseZap, Download, FileChartColumnIncreasing, X } from 'lucide-vue-next'
import { downloadAnalysisReport } from '../api/analysisReport'

const props = defineProps({ report: { type: Object, required: true } })
defineEmits(['close'])

function download() { downloadAnalysisReport(props.report) }
function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-'
  return typeof value === 'object' ? JSON.stringify(value) : String(value)
}
</script>

<style scoped>
.report-backdrop{position:fixed;inset:0;z-index:1200;padding:18px;background:rgba(12,22,37,.7);backdrop-filter:blur(9px)}
.report-window{width:min(1180px,100%);height:100%;margin:auto;overflow:hidden;border:1px solid rgba(255,255,255,.22);border-radius:24px;background:#f1f4f7;box-shadow:0 30px 90px rgba(4,13,27,.4)}
.report-toolbar{height:62px;display:flex;align-items:center;justify-content:space-between;padding:0 20px;color:#17243a;background:#fff;border-bottom:1px solid #dfe5ec}
.report-toolbar>div,.report-toolbar nav,.report-toolbar button{display:flex;align-items:center;gap:9px}.report-toolbar>div{font-weight:800}.report-toolbar nav{gap:8px}.report-toolbar button{height:38px;padding:0 12px;border:1px solid #dce3eb;border-radius:10px;color:#334258;background:#fff;cursor:pointer}.report-toolbar .report-download{color:#fff;border-color:#167d6d;background:#167d6d;font-weight:700}
.report-scroll{height:calc(100% - 62px);overflow:auto}.report-page{width:min(1060px,calc(100% - 40px));margin:28px auto 60px}.report-hero{position:relative;overflow:hidden;padding:46px;border-radius:27px;color:#fff;background:linear-gradient(135deg,#15243e,#203e67 65%,#27618a);box-shadow:0 24px 64px rgba(22,43,72,.2)}.report-hero:after{content:"";position:absolute;width:270px;height:270px;right:-72px;top:-130px;border-radius:50%;background:rgba(78,218,185,.18)}.report-brand{color:#75dfc1;font-size:12px;letter-spacing:.13em}.report-hero h1{max-width:780px;margin:18px 0 12px;font-size:35px;line-height:1.2}.report-hero p{max-width:810px;margin:0;color:#d9e5f2;line-height:1.75}.report-meta{display:flex;flex-wrap:wrap;gap:18px;margin-top:25px;color:#bccce0;font-size:13px}.report-meta span{display:flex;align-items:center;gap:6px}
.report-metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin:16px 0}.report-metrics article,.report-panel{border:1px solid #e0e6ed;border-radius:19px;background:#fff;box-shadow:0 10px 35px rgba(38,55,77,.06)}.report-metrics article{padding:21px}.report-metrics span,.report-metrics small{display:block;color:#728095;font-size:12px}.report-metrics strong{display:block;margin:10px 0 7px;color:#15243e;font-size:26px}.report-panel{padding:25px;margin-top:15px}.report-panel-title>span{color:#1c8975;font-size:11px;letter-spacing:.12em}.report-panel-title h2{margin:5px 0 18px;color:#1b293d;font-size:20px}
.report-bars{height:215px;display:flex;align-items:flex-end;gap:10px;padding:15px 4px 0}.report-bar-item{flex:1;min-width:40px;text-align:center}.report-bar-track{height:140px;display:flex;align-items:flex-end;overflow:hidden;border-radius:8px;background:#edf3f7}.report-bar-track i{display:block;width:100%;min-height:4px;border-radius:8px 8px 0 0;background:linear-gradient(180deg,#4bceb0,#2584a7)}.report-bar-item b,.report-bar-item span{display:block;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}.report-bar-item b{margin-top:6px;font-size:11px}.report-bar-item span{margin-top:3px;color:#7e8b9d;font-size:11px}
.report-two-column{display:grid;grid-template-columns:1fr 1fr;gap:15px}.report-list{display:grid;gap:9px}.report-list>div{display:grid;grid-template-columns:auto 1fr;gap:11px;align-items:start;padding:13px;border-radius:13px;background:#f5faf8}.report-list b{min-width:20px;color:#147d6b}.report-list p{margin:0;color:#435167;line-height:1.55}.risk-panel .report-list>div{background:#fff7f1}.risk-panel .report-list b{color:#bd5c32}
.report-table-scroll{overflow:auto}.report-table-panel table{width:100%;border-collapse:collapse;font-size:13px}.report-table-panel th,.report-table-panel td{padding:11px 13px;text-align:left;white-space:nowrap;border-bottom:1px solid #e8edf2}.report-table-panel th{color:#65748a;background:#f7f9fb}.report-definitions{display:grid;grid-template-columns:repeat(2,1fr);gap:11px}.report-definitions>div{padding:14px;border-radius:13px;background:#f6f8fb}.report-definitions p{margin:6px 0 0;color:#556277;font-size:13px;line-height:1.55}.report-foot{margin-top:16px;padding:20px;color:#6a788c;font-size:12px;line-height:1.65}.report-foot p{margin:5px 0}.report-foot b{color:#334157}
@media(max-width:760px){.report-backdrop{padding:0}.report-window{border-radius:0}.report-page{width:min(100% - 18px,1060px);margin-top:9px}.report-hero{padding:29px 23px}.report-hero h1{font-size:27px}.report-metrics{grid-template-columns:repeat(2,1fr)}.report-two-column,.report-definitions{grid-template-columns:1fr}.report-panel{padding:19px}.report-toolbar .report-download{font-size:0}.report-toolbar .report-download svg{margin:0}}
</style>
