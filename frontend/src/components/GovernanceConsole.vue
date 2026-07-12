<template>
  <div class="governance-backdrop" @click.self="$emit('close')">
    <section class="governance-console">
      <header>
        <div><span>内部经营配置</span><h2>数据与分析管理台</h2><p>面向运营管理员，不影响普通商家的提问体验。</p></div>
        <button type="button" class="close" @click="$emit('close')"><X :size="20" /></button>
      </header>
      <nav>
        <button v-for="item in tabs" :key="item.id" :class="{ active: tab === item.id }" @click="tab = item.id">
          <component :is="item.icon" :size="16" />{{ item.label }}
        </button>
      </nav>

      <div v-if="loading" class="console-state"><LoaderCircle :size="20" />正在读取配置</div>
      <div v-else-if="error" class="console-state error">{{ error }}</div>

      <section v-else-if="tab === 'knowledge'" class="console-content">
        <div class="section-title"><div><h3>经营知识建议</h3><p>来自商家纠正与反馈，审核后才能进入正式口径。</p></div><button @click="loadKnowledge">刷新</button></div>
        <div v-if="!suggestions.length" class="empty">暂无待处理建议</div>
        <article v-for="item in suggestions" :key="item.suggestionId" class="governance-item">
          <div><b>{{ item.metricName || item.payload?.title || item.suggestionType || '经营知识建议' }}</b><span>{{ statusLabel(item.status) }}</span></div>
          <p>{{ item.payload?.content || item.payload?.correctionText || item.reviewNote || '等待运营人员补充说明' }}</p>
          <div class="publish-targets">
            <input v-model="item.topic" placeholder="经营主题，例如 电商交易" />
            <input v-model="item.sourceTable" placeholder="目标数据表" />
          </div>
          <footer>
            <button @click="review(item, false)">舍弃</button>
            <button class="secondary" @click="review(item, true)">审核通过</button>
            <button class="primary" :disabled="!publishable(item)" @click="publish(item)">发布并更新索引</button>
          </footer>
        </article>
      </section>

      <section v-else-if="tab === 'assets'" class="console-content">
        <div class="section-title"><div><h3>数据资产建设</h3><p>从真实表结构和样例数据生成业务字段、指标和规则候选。</p></div></div>
        <form class="builder-form" @submit.prevent="buildAsset">
          <label>经营主题<select v-model="builder.topic"><option v-for="name in topics" :key="name">{{ name }}</option></select></label>
          <label>数据表<input v-model="builder.tableName" placeholder="例如 ads_merchant_profile" /></label>
          <label class="wide">业务说明<textarea v-model="builder.businessKnowledge" rows="4" placeholder="补充这张表在商家经营中的用途和关键口径" /></label>
          <button class="primary" :disabled="!builder.topic || !builder.tableName">生成待审核资产</button>
        </form>
        <pre v-if="buildResult">{{ JSON.stringify(buildResult, null, 2) }}</pre>
        <div class="asset-editor-head">
          <div><h3>在线审核与发布</h3><p>编辑指标、字段和强制经营规则草稿，预检后再发布。</p></div>
          <div class="asset-picker">
            <select v-model="assetTopic" @change="loadAssetTables"><option value="">选择主题</option><option v-for="name in topics" :key="name">{{ name }}</option></select>
            <select v-model="assetTable" @change="loadGovernance"><option value="">选择数据表</option><option v-for="name in assetTables" :key="name">{{ name }}</option></select>
          </div>
        </div>
        <div v-if="assetGovernance" class="asset-workbench">
          <div class="asset-actions">
            <button @click="saveDraft">保存待审核草稿</button>
            <button class="secondary" @click="loadGovernance">重新载入</button>
            <button class="primary" @click="publishAsset">预检并发布</button>
          </div>
          <textarea v-model="assetJson" class="json-editor" spellcheck="false" />
          <div class="governance-summary">
            <article><b>影响检查</b><span>{{ assetGovernance.impact?.impactCount || 0 }} 个指标受影响</span><pre>{{ JSON.stringify(assetGovernance.impact?.schemaDriftReport || {}, null, 2) }}</pre></article>
            <article><b>待发布差异</b><span>{{ assetGovernance.pendingPatch?.changes?.length || 0 }} 项变化</span><pre>{{ JSON.stringify(assetGovernance.pendingPatch || assetGovernance.pendingAsset || {}, null, 2) }}</pre></article>
          </div>
          <div class="history-list"><h4>发布与回滚记录</h4><article v-for="(record, index) in historyItems" :key="index"><span>{{ record.status || 'PUBLISHED' }}</span><b>{{ record.semanticVersion || record.semanticCatalogVersion?.semanticVersion || '版本记录' }}</b><small>{{ record.publishedAt || record.rolledBackAt || record.createdAt }}</small><button v-if="record.semanticVersion" @click="rollbackAsset(record.semanticVersion)">回滚到此版本</button></article></div>
        </div>
      </section>

      <section v-else class="console-content">
        <div class="section-title"><div><h3>专项分析方案</h3><p>普通商家只会看到“经营体检、原因诊断”等业务动作。</p></div><button @click="loadCatalog">刷新</button></div>
        <div class="plan-grid">
          <article v-for="item in catalog" :key="item.skillName" class="plan-card">
            <b>{{ planLabel(item.skillName) }}</b><p>{{ item.displayName || item.skillName }}</p><span>{{ item.status || 'available' }}</span>
            <button class="primary" @click="install(item)">启用分析方案</button>
          </article>
        </div>
      </section>
    </section>
  </div>
</template>

<script setup>
import { computed, markRaw, onMounted, ref, watch } from 'vue'
import { BookOpenCheck, Boxes, LoaderCircle, Workflow, X } from 'lucide-vue-next'
import { buildTopicAsset, getAnalysisCatalog, getKnowledgeSuggestions, getTopicAssets, getTopicTableGovernance, getTopics, installAnalysisPlan, publishKnowledgeSuggestion, publishTopicTable, reviewKnowledgeSuggestion, rollbackTopicTable, saveTopicTableDraft } from '../api/client'

defineEmits(['close'])
const tabs = [
  { id: 'knowledge', label: '经营知识', icon: markRaw(BookOpenCheck) },
  { id: 'assets', label: '数据资产', icon: markRaw(Boxes) },
  { id: 'plans', label: '分析方案', icon: markRaw(Workflow) }
]
const tab = ref('knowledge')
const loading = ref(false)
const error = ref('')
const suggestions = ref([])
const topics = ref([])
const catalog = ref([])
const buildResult = ref(null)
const builder = ref({ topic: '', tableName: '', businessKnowledge: '' })
const assetTopic = ref('')
const assetTable = ref('')
const assetTables = ref([])
const assetGovernance = ref(null)
const assetJson = ref('')
const historyItems = computed(() => Array.isArray(assetGovernance.value?.publishHistory) ? assetGovernance.value.publishHistory : (assetGovernance.value?.publishHistory?.items || []))

onMounted(loadKnowledge)
watch(tab, value => { if (value === 'assets') loadTopics(); if (value === 'plans') loadCatalog() })

async function loadKnowledge() { await run(async () => { suggestions.value = (await getKnowledgeSuggestions()).items || [] }) }
async function loadTopics() { await run(async () => { topics.value = (await getTopics()).items || []; builder.value.topic ||= topics.value[0] || ''; assetTopic.value ||= topics.value[0] || ''; await loadAssetTables(false) }) }
async function loadCatalog() { await run(async () => { catalog.value = (await getAnalysisCatalog()).items || [] }) }
async function review(item, approved) {
  await run(async () => { await reviewKnowledgeSuggestion(item.suggestionId, { approved, action: approved ? 'approve' : 'reject', reviewer: 'merchant_ops' }); await loadKnowledge() })
}
async function publish(item) {
  await run(async () => { await publishKnowledgeSuggestion(item.suggestionId, { reviewer: 'merchant_ops', topic: item.topic, tableName: item.sourceTable, autoIndex: true }); await loadKnowledge() })
}
async function buildAsset() {
  await run(async () => { buildResult.value = await buildTopicAsset({ topic: builder.value.topic, tableName: builder.value.tableName, merchantId: '100', businessKnowledge: builder.value.businessKnowledge }) })
}
async function loadAssetTables(withLoading = true) {
  const action = async () => {
    if (!assetTopic.value) return
    const files = (await getTopicAssets(assetTopic.value)).items || []
    assetTables.value = [...new Set(files.map(path => path.match(/^tables\/([^/]+)\//)?.[1]).filter(Boolean))]
    if (!assetTables.value.includes(assetTable.value)) assetTable.value = assetTables.value[0] || ''
    if (assetTable.value) await loadGovernance(false)
  }
  if (withLoading) await run(action); else await action()
}
async function loadGovernance(withLoading = true) {
  const action = async () => {
    if (!assetTopic.value || !assetTable.value) return
    assetGovernance.value = await getTopicTableGovernance(assetTopic.value, assetTable.value)
    const draft = assetGovernance.value.pendingAsset && Object.keys(assetGovernance.value.pendingAsset).length ? assetGovernance.value.pendingAsset : assetGovernance.value.asset
    assetJson.value = JSON.stringify({
      description: draft.description || '',
      metrics: draft.metrics || [],
      terms: draft.terms || [],
      knowledgeRules: draft.knowledgeRules || []
    }, null, 2)
  }
  if (withLoading) await run(action); else await action()
}
async function saveDraft() { await run(async () => { await saveTopicTableDraft(assetTopic.value, assetTable.value, { ...JSON.parse(assetJson.value), editor: 'merchant_ops' }); await loadGovernance(false) }) }
async function publishAsset() { await run(async () => { await saveTopicTableDraft(assetTopic.value, assetTable.value, { ...JSON.parse(assetJson.value), editor: 'merchant_ops' }); const result = await publishTopicTable(assetTopic.value, assetTable.value, { approved: true, reviewer: 'merchant_ops', reviewNote: '内部管理台审核发布' }); if (!result.success) throw new Error(result.status || '预检未通过'); await loadGovernance(false) }) }
async function rollbackAsset(version) { await run(async () => { const result = await rollbackTopicTable(assetTopic.value, assetTable.value, version); if (!result.success) throw new Error(result.status || '回滚失败'); await loadGovernance(false) }) }
async function install(item) { await run(async () => { await installAnalysisPlan(item.skillName, { scope: 'merchant', merchantIds: ['100'], trafficPercent: 100 }); await loadCatalog() }) }
async function run(fn) { loading.value = true; error.value = ''; try { await fn() } catch (e) { error.value = `操作失败：${e.message || e}` } finally { loading.value = false } }
function publishable(item) { return ['approved', 'publish_requested', 'published', 'indexed'].includes(String(item.status || '').toLowerCase()) && item.topic && item.sourceTable }
function statusLabel(value) { return ({ candidate: '待审核', approved: '已通过', rejected: '已舍弃', published: '已发布', indexed: '已生效' })[String(value || '').toLowerCase()] || value || '待处理' }
function planLabel(name) { return ({ bi_trend_attribution: '指标波动原因深挖', gmv_drop_diagnosis: 'GMV下降原因诊断', merchant_daily_briefing: '店铺经营体检', new_product_risk: '新品经营风险排查', ratio_analysis: '占比口径核验', refund_rate_diagnosis: '退款压力专项诊断', risk_analysis: '经营风险优先级分析', rule_compliance: '平台规则影响核对' })[name] || '经营专项分析' }
</script>

<style scoped>
.governance-backdrop{position:fixed;inset:0;z-index:80;background:rgba(15,23,42,.42);display:grid;place-items:center;padding:24px}.governance-console{width:min(980px,96vw);height:min(760px,92vh);overflow:hidden;background:#f8fbff;border-radius:22px;box-shadow:0 28px 80px rgba(15,23,42,.28);display:grid;grid-template-rows:auto auto 1fr}.governance-console>header{display:flex;justify-content:space-between;padding:22px 24px 16px;background:#fff}.governance-console h2,.governance-console h3,.governance-console p{margin:0}.governance-console header span{color:#4779d6;font-size:12px;font-weight:800}.governance-console header p,.section-title p{margin-top:5px;color:#778195;font-size:13px}.close{background:#eef3fa;border-radius:10px;width:38px;height:38px}.governance-console nav{display:flex;gap:8px;padding:10px 24px;background:#fff;border-top:1px solid #eef2f7;border-bottom:1px solid #e5ebf4}.governance-console nav button{display:flex;gap:7px;align-items:center;padding:9px 14px;border-radius:10px;background:transparent;color:#64748b}.governance-console nav button.active{background:#eaf2ff;color:#2563eb;font-weight:800}.console-content{overflow:auto;padding:22px 24px}.section-title{display:flex;justify-content:space-between;margin-bottom:16px}.section-title button,.governance-item button,.plan-card button{padding:8px 12px;border-radius:9px}.governance-item{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px;margin-bottom:12px}.governance-item>div{display:flex;justify-content:space-between}.governance-item span{color:#2563eb;font-size:12px}.governance-item p{margin:10px 0;color:#475569}.governance-item footer{display:flex;justify-content:flex-end;gap:8px;margin-top:14px}.publish-targets{display:grid!important;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.publish-targets input{min-width:0;border:1px solid #dbe3ef;border-radius:9px;padding:9px;background:#fbfdff}.primary{background:#2563eb!important;color:white}.secondary{background:#eaf2ff!important;color:#2563eb}.builder-form{display:grid;grid-template-columns:1fr 1fr;gap:14px;background:#fff;padding:18px;border-radius:14px}.builder-form label{display:grid;gap:7px;font-size:13px;font-weight:700}.builder-form .wide{grid-column:1/-1}.builder-form input,.builder-form select,.builder-form textarea{border:1px solid #dbe3ef;border-radius:9px;padding:10px;background:#fff}.builder-form button{width:max-content;padding:10px 16px;border-radius:10px}.plan-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.plan-card{display:grid;gap:9px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px}.plan-card span{font-size:12px;color:#64748b}.console-state{display:flex;align-items:center;justify-content:center;gap:8px}.error{color:#b91c1c}.empty{padding:40px;text-align:center;color:#94a3b8}pre{white-space:pre-wrap;background:#0f172a;color:#dbeafe;padding:14px;border-radius:12px;max-height:260px;overflow:auto}@media(max-width:760px){.plan-grid{grid-template-columns:1fr}.builder-form{grid-template-columns:1fr}.governance-backdrop{padding:0}.governance-console{width:100vw;height:100vh;border-radius:0}}
.asset-editor-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:26px 0 12px}.asset-editor-head p{color:#778195;font-size:13px;margin-top:5px}.asset-picker{display:flex;gap:8px}.asset-picker select{max-width:210px;border:1px solid #dbe3ef;border-radius:9px;padding:9px;background:#fff}.asset-workbench{display:grid;gap:12px}.asset-actions{display:flex;justify-content:flex-end;gap:8px}.asset-actions button,.history-list button{padding:8px 12px;border-radius:9px}.json-editor{box-sizing:border-box;width:100%;min-height:320px;resize:vertical;border:1px solid #233450;border-radius:12px;padding:14px;background:#0f172a;color:#dbeafe;font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.governance-summary{display:grid;grid-template-columns:1fr 1fr;gap:12px}.governance-summary article{min-width:0;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:12px}.governance-summary article>span{float:right;color:#64748b;font-size:12px}.governance-summary pre{max-height:180px}.history-list{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px}.history-list article{display:grid;grid-template-columns:100px 1fr 180px auto;align-items:center;gap:10px;padding:9px 0;border-top:1px solid #edf2f7}.history-list span,.history-list small{color:#64748b;font-size:12px}@media(max-width:760px){.asset-editor-head{align-items:stretch;flex-direction:column}.asset-picker{flex-direction:column}.governance-summary{grid-template-columns:1fr}.history-list article{grid-template-columns:1fr 1fr}.json-editor{min-height:240px}}
</style>
