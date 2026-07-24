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
          <section v-if="conflictReport(item)?.status === 'confirmation_required'" class="conflict-review">
            <div>
              <b>发现相似或冲突的公共知识</b>
              <span>报告 {{ shortReportId(conflictReport(item)?.reportId) }}</span>
            </div>
            <p>{{ conflictReport(item)?.message || '必须先确认处理方式，才能审核和发布。' }}</p>
            <p v-if="item.conflictError" class="inline-error">{{ item.conflictError }}</p>
            <ul>
              <li v-for="match in conflictReport(item)?.matches || []" :key="match.existingKnowledgeId">
                <b>{{ match.title || match.existingKnowledgeId }}</b>
                <span>{{ relationLabel(match.relation) }} · 相似度 {{ similarityPercent(match.similarity) }}</span>
                <p>{{ match.existingText }}</p>
                <small>{{ match.reason }}</small>
              </li>
            </ul>
            <div class="conflict-resolution">
              <label>
                处理方式
                <select v-model="item.conflictResolution">
                  <option value="">请选择</option>
                  <option v-for="option in conflictReport(item)?.resolutionOptions || []" :key="option" :value="option">
                    {{ resolutionLabel(option) }}
                  </option>
                </select>
              </label>
              <label v-if="item.conflictResolution === 'merge'" class="wide">
                融合后的正式内容
                <textarea v-model="item.mergedContent" rows="3" placeholder="请填写融合后唯一生效的知识内容" />
              </label>
              <button
                class="primary"
                :disabled="!resolutionReady(item)"
                @click="review(item, true, true)"
              >
                确认处理并审核
              </button>
            </div>
          </section>
          <footer>
            <button @click="review(item, false)">舍弃</button>
            <button @click="precheck(item)">冲突预检</button>
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
          <section class="review-workflow-card">
            <div class="review-workflow-head">
              <div><span>审核状态</span><b>{{ workflowStatusLabel }}</b></div>
              <small v-if="reviewWorkflow.submittedBy">提交人：{{ reviewWorkflow.submittedBy }}</small>
              <small v-if="reviewWorkflow.reviewedBy">审核人：{{ reviewWorkflow.reviewedBy }}</small>
              <small v-if="reviewWorkflow.publishedBy">发布人：{{ reviewWorkflow.publishedBy }}</small>
            </div>
            <div class="review-steps">
              <span :class="{ active: workflowStep >= 1 }">1 保存草稿</span>
              <span :class="{ active: workflowStep >= 2 }">2 提交审核</span>
              <span :class="{ active: workflowStep >= 3 }">3 审核通过</span>
              <span :class="{ active: workflowStep >= 4 }">4 发布生效</span>
            </div>
            <label>
              审核说明
              <input v-model="reviewNote" placeholder="填写修改原因、核查结果或发布说明" />
            </label>
            <div class="asset-actions">
              <button @click="saveDraft">{{ workflowStatus === 'DRAFT' ? '保存草稿' : '保存修改并重置审核' }}</button>
              <button class="secondary" @click="loadGovernance">重新载入</button>
              <button v-if="['DRAFT', 'REJECTED'].includes(workflowStatus)" class="primary" @click="submitAssetReview">提交审核</button>
              <button v-if="workflowStatus === 'PENDING_REVIEW'" class="danger" @click="reviewAsset(false)">驳回</button>
              <button v-if="workflowStatus === 'PENDING_REVIEW'" class="primary" @click="reviewAsset(true)">审核通过</button>
              <button v-if="workflowStatus === 'APPROVED'" class="primary" @click="publishAsset">发布并更新索引</button>
            </div>
          </section>
          <section class="semantic-editor">
            <header class="semantic-editor-summary">
              <div>
                <span>结构化语义编辑</span>
                <h4>{{ assetTable }}</h4>
                <p>直接维护业务字段、指标公式、表关系和经营规则，无需编辑 JSON。</p>
              </div>
              <div class="change-totals">
                <span class="change-total added">{{ diffStats.added }} 新增</span>
                <span class="change-total changed">{{ diffStats.changed }} 修改</span>
                <span class="change-total removed">{{ diffStats.removed }} 删除</span>
              </div>
            </header>

            <nav class="semantic-section-nav" aria-label="语义资产分类">
              <button
                v-for="section in semanticSections"
                :key="section.id"
                type="button"
                :class="{ active: semanticSection === section.id }"
                @click="selectSemanticSection(section.id)"
              >
                <component :is="section.icon" :size="16" />
                <span>{{ section.label }}</span>
                <b>{{ section.count }}</b>
              </button>
            </nav>

            <div v-if="semanticSection === 'description'" class="description-editor">
              <label>
                表业务说明
                <textarea
                  v-model="assetDraft.description"
                  rows="7"
                  placeholder="说明这张表的业务用途、数据粒度和使用边界"
                />
              </label>
              <aside>
                <b>填写建议</b>
                <p>说明“一行代表什么”、适合回答哪些问题，以及不适合怎样使用。</p>
              </aside>
            </div>

            <div v-else class="semantic-editor-body">
              <aside class="semantic-item-list">
                <div class="semantic-list-tools">
                  <label>
                    <Search :size="15" />
                    <input v-model="semanticSearch" :placeholder="`搜索${activeSectionLabel}`" />
                  </label>
                  <button type="button" class="icon-button" :title="`新增${activeSectionLabel}`" @click="addSemanticItem">
                    <Plus :size="17" />
                  </button>
                </div>
                <div v-if="!filteredSemanticItems.length" class="semantic-list-empty">
                  暂无{{ activeSectionLabel }}
                  <button type="button" @click="addSemanticItem">立即新增</button>
                </div>
                <button
                  v-for="entry in filteredSemanticItems"
                  :key="`${semanticSection}-${entry.index}`"
                  type="button"
                  class="semantic-list-item"
                  :class="{ active: selectedSemanticIndex === entry.index }"
                  @click="selectedSemanticIndex = entry.index"
                >
                  <span>{{ semanticItemTitle(entry.item, entry.index) }}</span>
                  <small>{{ semanticItemSubtitle(entry.item) }}</small>
                  <i :class="itemDiffStatus(semanticSection, entry.item)">{{ diffStatusLabel(itemDiffStatus(semanticSection, entry.item)) }}</i>
                </button>
              </aside>

              <section v-if="selectedSemanticItem" class="semantic-form">
                <header>
                  <div><span>{{ activeSectionLabel }}详情</span><b>{{ semanticItemTitle(selectedSemanticItem, selectedSemanticIndex) }}</b></div>
                  <button type="button" class="delete-button" @click="removeSemanticItem">
                    <Trash2 :size="15" />删除
                  </button>
                </header>

                <div v-if="semanticSection === 'semanticColumns'" class="form-grid">
                  <label>物理字段名<input v-model="selectedSemanticItem.columnName" placeholder="例如 order_id" /></label>
                  <label>业务名称<input v-model="selectedSemanticItem.businessName" placeholder="例如 主订单号" /></label>
                  <label>字段角色<select v-model="selectedSemanticItem.role"><option v-for="role in fieldRoles" :key="role" :value="role">{{ role }}</option></select></label>
                  <label>比较策略<select v-model="selectedSemanticItem.comparisonPolicy"><option v-for="policy in comparisonPolicies" :key="policy" :value="policy">{{ policy }}</option></select></label>
                  <label class="wide">业务说明<textarea v-model="selectedSemanticItem.description" rows="3" placeholder="这个字段的业务含义和使用限制" /></label>
                  <label class="wide">别名<input :value="listText(selectedSemanticItem.aliases)" @input="updateListField(selectedSemanticItem, 'aliases', $event)" placeholder="多个值用逗号分隔" /></label>
                  <label>标准实体引用<input v-model="selectedSemanticItem.canonicalEntityRef" placeholder="例如 entity:order" /></label>
                  <label>允许的筛选符<input :value="listText(selectedSemanticItem.filterOperators)" @input="updateListField(selectedSemanticItem, 'filterOperators', $event)" placeholder="EQ, IN" /></label>
                </div>

                <div v-else-if="semanticSection === 'metrics'" class="form-grid">
                  <label>指标编码<input v-model="selectedSemanticItem.metricKey" placeholder="例如 refund_rate" /></label>
                  <label>指标名称<input v-model="selectedSemanticItem.businessName" placeholder="例如 退款率" /></label>
                  <label class="wide formula-field">计算公式<textarea v-model="selectedSemanticItem.formula" rows="3" spellcheck="false" placeholder="例如 SUM(refund_amount) / NULLIF(SUM(pay_amount), 0)" /></label>
                  <label>单位<input v-model="selectedSemanticItem.unit" placeholder="例如 %、元、单" /></label>
                  <label>时间字段<input v-model="selectedSemanticItem.timeColumn" placeholder="例如 pt" /></label>
                  <label>指标层级<select v-model="selectedSemanticItem.metricLevel"><option value="atomic">原子指标</option><option value="derived">派生指标</option><option value="business">业务指标</option><option value="composite">复合指标</option></select></label>
                  <label>聚合策略<input v-model="selectedSemanticItem.aggregationPolicy" placeholder="例如 period_rollup" /></label>
                  <label>统计粒度<input v-model="selectedSemanticItem.metricGrain" placeholder="例如 order_detail" /></label>
                  <label class="wide">来源字段<input :value="listText(selectedSemanticItem.sourceColumns)" @input="updateListField(selectedSemanticItem, 'sourceColumns', $event)" placeholder="多个值用逗号分隔" /></label>
                  <label class="wide">业务说明<textarea v-model="selectedSemanticItem.description" rows="3" /></label>
                  <label class="wide">别名<input :value="listText(selectedSemanticItem.aliases)" @input="updateListField(selectedSemanticItem, 'aliases', $event)" placeholder="退款占比, refund rate" /></label>
                </div>

                <div v-else-if="semanticSection === 'relationships'" class="form-grid">
                  <label>关系编码<input v-model="selectedSemanticItem.name" placeholder="例如 order_refund_by_sub_order" /></label>
                  <label>连接类型<select v-model="selectedSemanticItem.joinType"><option value="LEFT">LEFT</option><option value="INNER">INNER</option><option value="RIGHT">RIGHT</option><option value="FULL">FULL</option></select></label>
                  <label>左表<select v-model="selectedSemanticItem.leftTable"><option v-for="name in assetTables" :key="name" :value="name">{{ name }}</option></select></label>
                  <label>右表<select v-model="selectedSemanticItem.rightTable"><option v-for="name in assetTables" :key="name" :value="name">{{ name }}</option></select></label>
                  <label>基数关系<select v-model="selectedSemanticItem.cardinality"><option value="one_to_one">一对一</option><option value="one_to_many">一对多</option><option value="many_to_one">多对一</option><option value="many_to_many">多对多</option></select></label>
                  <label>防膨胀策略<input v-model="selectedSemanticItem.fanoutPolicy" placeholder="例如 DIRECTIONAL_GRAIN_GUARD" /></label>
                  <label class="wide">关系粒度<input v-model="selectedSemanticItem.grain" placeholder="例如 sub_order_id_refund_id" /></label>
                  <div class="wide relationship-keys">
                    <div><b>关联键</b><button type="button" @click="addRelationshipKey"><Plus :size="14" />添加一组</button></div>
                    <div v-for="(pair, pairIndex) in selectedSemanticItem.keys || []" :key="pairIndex" class="key-pair">
                      <input :value="pair[0]" placeholder="左表字段" @input="updateRelationshipKey(pairIndex, 0, $event)" />
                      <span>对应</span>
                      <input :value="pair[1]" placeholder="右表字段" @input="updateRelationshipKey(pairIndex, 1, $event)" />
                      <button type="button" title="删除关联键" @click="removeRelationshipKey(pairIndex)"><Trash2 :size="14" /></button>
                    </div>
                  </div>
                  <label class="wide">适用场景<input :value="listText(selectedSemanticItem.useCases)" @input="updateListField(selectedSemanticItem, 'useCases', $event)" placeholder="多个值用逗号分隔" /></label>
                  <label class="wide">使用警告<textarea :value="listText(selectedSemanticItem.cautions, '\\n')" rows="3" @input="updateListField(selectedSemanticItem, 'cautions', $event)" placeholder="每行一条风险说明" /></label>
                </div>

                <div v-else-if="semanticSection === 'terms'" class="form-grid">
                  <label>业务术语<input v-model="selectedSemanticItem.term" placeholder="例如 主订单号" /></label>
                  <label>关联字段<input :value="listText(selectedSemanticItem.relatedColumns)" @input="updateListField(selectedSemanticItem, 'relatedColumns', $event)" placeholder="order_id" /></label>
                  <label class="wide">术语解释<textarea v-model="selectedSemanticItem.description" rows="4" /></label>
                  <label class="wide">别名<input :value="listText(selectedSemanticItem.aliases)" @input="updateListField(selectedSemanticItem, 'aliases', $event)" placeholder="多个值用逗号分隔" /></label>
                </div>

                <div v-else class="form-grid">
                  <label>规则标题<input v-model="selectedSemanticItem.title" placeholder="例如 时间范围过滤" /></label>
                  <label class="switch-label"><input v-model="selectedSemanticItem.alwaysApply" type="checkbox" />所有查询强制应用</label>
                  <label class="wide">规则内容<textarea v-model="selectedSemanticItem.content" rows="5" placeholder="描述规则触发条件、约束和例外" /></label>
                  <label class="wide">触发关键词<input :value="listText(selectedSemanticItem.keywords)" @input="updateListField(selectedSemanticItem, 'keywords', $event)" placeholder="时间, 最近, pt" /></label>
                </div>
              </section>
              <section v-else class="semantic-form-empty">选择左侧资产，或新增一条{{ activeSectionLabel }}。</section>
            </div>
          </section>

          <section class="visual-diff">
            <header>
              <div><GitCompare :size="18" /><div><b>版本差异</b><span>当前已发布版本 → 本次草稿</span></div></div>
              <span>{{ diffEntries.length }} 项变化</span>
            </header>
            <div v-if="!diffEntries.length" class="diff-empty">当前草稿与已发布版本一致</div>
            <div v-else class="diff-list">
              <article v-for="entry in diffEntries" :key="`${entry.section}-${entry.identity}-${entry.status}`">
                <span :class="entry.status">{{ diffStatusLabel(entry.status) }}</span>
                <div><b>{{ entry.title }}</b><small>{{ sectionLabel(entry.section) }}</small></div>
                <p>{{ diffSummary(entry) }}</p>
                <div v-if="entry.status === 'changed'" class="diff-values">
                  <div v-for="field in entry.fields.slice(0, 2)" :key="field">
                    <em>{{ semanticFieldLabel(field) }}</em>
                    <del>{{ formatDiffValue(entry.before?.[field]) }}</del>
                    <span>→</span>
                    <ins>{{ formatDiffValue(entry.after?.[field]) }}</ins>
                  </div>
                </div>
              </article>
            </div>
          </section>

          <div class="governance-summary">
            <article>
              <b>影响检查</b>
              <span>{{ assetGovernance.impact?.impactCount || 0 }} 个指标受影响</span>
              <p>{{ impactSummary }}</p>
            </article>
            <article>
              <b>发布范围</b>
              <span>{{ totalAssetCount }} 项语义资产</span>
              <p>只发布当前 {{ assetTopic }} / {{ assetTable }} 的已审核草稿，并同步更新召回索引。</p>
            </article>
          </div>
          <div class="history-list"><h4>发布与回滚记录</h4><article v-for="(record, index) in historyItems" :key="index"><span>{{ record.status || 'PUBLISHED' }}</span><b>{{ record.semanticVersion || record.semanticCatalogVersion?.semanticVersion || '版本记录' }}</b><small>{{ record.publishedAt || record.rolledBackAt || record.createdAt }}</small><button v-if="record.semanticVersion" @click="rollbackAsset(record.semanticVersion)">回滚到此版本</button></article></div>
        </div>
      </section>

      <section v-else class="console-content">
        <div class="section-title"><div><h3>专项分析方案</h3><p>普通商家只会看到“经营体检、原因诊断”等业务动作。</p></div><button @click="loadCatalog">刷新</button></div>
        <div class="plan-grid">
          <article v-for="item in catalog" :key="item.skillName" class="plan-card">
            <b>{{ item.displayName || item.skillName }}</b><p>{{ item.description || '' }}</p><span>{{ item.status || 'available' }}</span>
            <button class="primary" @click="install(item)">启用分析方案</button>
          </article>
        </div>
      </section>
    </section>
  </div>
</template>

<script setup>
import { computed, markRaw, onMounted, ref, watch } from 'vue'
import { BookOpenCheck, BookText, Boxes, Database, FileText, GitCompare, Link2, LoaderCircle, Plus, Search, ShieldCheck, Sigma, Trash2, Workflow, X } from 'lucide-vue-next'
import { buildTopicAsset, checkKnowledgeSuggestionConflicts, getAnalysisCatalog, getKnowledgeSuggestions, getTopicAssets, getTopicTableGovernance, getTopics, installAnalysisPlan, publishKnowledgeSuggestion, publishTopicTable, reviewKnowledgeSuggestion, reviewTopicTable, rollbackTopicTable, saveTopicTableDraft, submitTopicTableReview } from '../api/client'
import { pathSegment } from '../utils/textParsing'

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
const assetDraft = ref(emptyAssetDraft())
const activeAssetSnapshot = ref(emptyAssetDraft())
const semanticSection = ref('semanticColumns')
const selectedSemanticIndex = ref(0)
const semanticSearch = ref('')
const reviewNote = ref('')
const fieldRoles = ['KEY', 'ENTITY', 'TIME', 'DIMENSION', 'MEASURE', 'ATTRIBUTE']
const comparisonPolicies = ['exact', 'case_insensitive', 'trimmed', 'trimmed_case_insensitive', 'integer', 'decimal']
const semanticSectionMeta = [
  { id: 'description', label: '表说明', icon: markRaw(FileText) },
  { id: 'semanticColumns', label: '业务字段', icon: markRaw(Database) },
  { id: 'metrics', label: '指标公式', icon: markRaw(Sigma) },
  { id: 'relationships', label: '表关系', icon: markRaw(Link2) },
  { id: 'terms', label: '业务术语', icon: markRaw(BookText) },
  { id: 'knowledgeRules', label: '经营规则', icon: markRaw(ShieldCheck) }
]
const semanticSections = computed(() => semanticSectionMeta.map(section => ({
  ...section,
  count: section.id === 'description' ? 1 : sectionItems(section.id).length
})))
const activeSectionLabel = computed(() => sectionLabel(semanticSection.value))
const selectedSemanticItem = computed(() => sectionItems(semanticSection.value)[selectedSemanticIndex.value] || null)
const filteredSemanticItems = computed(() => {
  const query = semanticSearch.value.trim().toLowerCase()
  return sectionItems(semanticSection.value)
    .map((item, index) => ({ item, index }))
    .filter(entry => !query || JSON.stringify(entry.item).toLowerCase().includes(query))
})
const diffEntries = computed(buildDiffEntries)
const diffStats = computed(() => diffEntries.value.reduce((result, item) => {
  result[item.status] += 1
  return result
}, { added: 0, changed: 0, removed: 0 }))
const totalAssetCount = computed(() => ['semanticColumns', 'metrics', 'relationships', 'terms', 'knowledgeRules']
  .reduce((total, section) => total + sectionItems(section).length, 0))
const impactSummary = computed(() => {
  const drift = assetGovernance.value?.impact?.schemaDriftReport || {}
  const missing = drift.missingLiveColumns || drift.missing_live_columns || []
  const changed = drift.typeChangedColumns || drift.type_changed_columns || []
  if (!missing.length && !changed.length) return '当前未发现物理字段缺失或类型变化。'
  return `发现 ${missing.length} 个缺失字段、${changed.length} 个类型变化，请在发布前确认影响。`
})
const historyItems = computed(() => Array.isArray(assetGovernance.value?.publishHistory) ? assetGovernance.value.publishHistory : (assetGovernance.value?.publishHistory?.items || []))
const reviewWorkflow = computed(() => assetGovernance.value?.reviewWorkflow || {})
const workflowStatus = computed(() => String(reviewWorkflow.value.status || 'DRAFT').toUpperCase())
const workflowStatusLabel = computed(() => ({
  DRAFT: '草稿',
  PENDING_REVIEW: '待审核',
  APPROVED: '审核通过，待发布',
  REJECTED: '已驳回',
  PUBLISHED: '已发布'
}[workflowStatus.value] || workflowStatus.value))
const workflowStep = computed(() => ({
  DRAFT: 1,
  REJECTED: 1,
  PENDING_REVIEW: 2,
  APPROVED: 3,
  PUBLISHED: 4
}[workflowStatus.value] || 1))

onMounted(loadKnowledge)
watch(tab, value => { if (value === 'assets') loadTopics(); if (value === 'plans') loadCatalog() })

async function loadKnowledge() { await run(async () => { suggestions.value = ((await getKnowledgeSuggestions()).items || []).filter(item => !['merchant_active', 'dismissed'].includes(String(item.status || '').toLowerCase())) }) }
async function loadTopics() { await run(async () => { topics.value = (await getTopics()).items || []; builder.value.topic ||= topics.value[0] || ''; assetTopic.value ||= topics.value[0] || ''; await loadAssetTables(false) }) }
async function loadCatalog() { await run(async () => { catalog.value = (await getAnalysisCatalog()).items || [] }) }
async function precheck(item) {
  await run(async () => {
    const result = await checkKnowledgeSuggestionConflicts(item.suggestionId)
    if (!result.success) throw new Error(statusLabel(result.status))
    applyConflictResult(item, result)
    if (result.status === 'CONFLICT_CHECK_CLEAR') await loadKnowledge()
  })
}
async function review(item, approved, resolveConflict = false) {
  await run(async () => {
    const report = conflictReport(item)
    const result = await reviewKnowledgeSuggestion(item.suggestionId, {
      approved,
      action: approved ? 'approve' : 'reject',
      conflictResolution: resolveConflict ? item.conflictResolution || '' : '',
      conflictReportId: resolveConflict ? report?.reportId || '' : '',
      mergedContent: resolveConflict ? item.mergedContent || '' : ''
    })
    if (result.status === 'CONFLICT_CONFIRMATION_REQUIRED' || result.status === 'STALE_CONFLICT_REPORT' || result.status === 'MERGED_CONTENT_REQUIRED') {
      applyConflictResult(item, result)
      item.conflictError = result.status === 'CONFLICT_CONFIRMATION_REQUIRED' ? '' : statusLabel(result.status)
      return
    }
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadKnowledge()
  })
}
async function publish(item) {
  await run(async () => {
    const result = await publishKnowledgeSuggestion(item.suggestionId, { topic: item.topic, tableName: item.sourceTable, autoIndex: true })
    if (result.status === 'CONFLICT_CONFIRMATION_REQUIRED') {
      applyConflictResult(item, result)
      return
    }
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadKnowledge()
  })
}
async function buildAsset() {
  await run(async () => { buildResult.value = await buildTopicAsset({ topic: builder.value.topic, tableName: builder.value.tableName, businessKnowledge: builder.value.businessKnowledge }) })
}
async function loadAssetTables(withLoading = true) {
  const action = async () => {
    if (!assetTopic.value) return
    const files = (await getTopicAssets(assetTopic.value)).items || []
    assetTables.value = [...new Set(files.map(path => pathSegment(path, 'tables/', 0)).filter(Boolean))]
    if (!assetTables.value.includes(assetTable.value)) assetTable.value = assetTables.value[0] || ''
    if (assetTable.value) await loadGovernance(false)
  }
  if (withLoading) await run(action); else await action()
}
async function loadGovernance(withLoading = true) {
  const action = async () => {
    if (!assetTopic.value || !assetTable.value) return
    assetGovernance.value = await getTopicTableGovernance(assetTopic.value, assetTable.value)
    reviewNote.value = assetGovernance.value.reviewWorkflow?.reviewNote || assetGovernance.value.reviewWorkflow?.submissionNote || ''
    const active = assetGovernance.value.asset || {}
    const draft = assetGovernance.value.pendingAsset && Object.keys(assetGovernance.value.pendingAsset).length ? assetGovernance.value.pendingAsset : active
    activeAssetSnapshot.value = normalizedAssetDraft(active, assetGovernance.value.relationships)
    assetDraft.value = normalizedAssetDraft(draft, assetGovernance.value.pendingRelationships)
    selectedSemanticIndex.value = 0
    semanticSearch.value = ''
  }
  if (withLoading) await run(action); else await action()
}
async function saveDraft() {
  await run(async () => {
    const result = await saveTopicTableDraft(assetTopic.value, assetTable.value, cloneValue(assetDraft.value))
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadGovernance(false)
  })
}
async function submitAssetReview() {
  await run(async () => {
    const result = await submitTopicTableReview(assetTopic.value, assetTable.value, { reviewNote: reviewNote.value })
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadGovernance(false)
  })
}
async function reviewAsset(approved) {
  await run(async () => {
    const result = await reviewTopicTable(assetTopic.value, assetTable.value, { approved, reviewNote: reviewNote.value })
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadGovernance(false)
  })
}
async function publishAsset() {
  await run(async () => {
    const result = await publishTopicTable(assetTopic.value, assetTable.value, { approved: true, reviewNote: reviewNote.value })
    if (!result.success) throw new Error(statusLabel(result.status))
    await loadGovernance(false)
  })
}
async function rollbackAsset(version) { await run(async () => { const result = await rollbackTopicTable(assetTopic.value, assetTable.value, version); if (!result.success) throw new Error(result.status || '回滚失败'); await loadGovernance(false) }) }
async function install(item) { await run(async () => { await installAnalysisPlan(item.skillName, item.installDefaults || {}); await loadCatalog() }) }
async function run(fn) { loading.value = true; error.value = ''; try { await fn() } catch (e) { error.value = `操作失败：${e.message || e}` } finally { loading.value = false } }
function emptyAssetDraft() {
  return {
    description: '',
    semanticColumns: [],
    metrics: [],
    relationships: [],
    terms: [],
    knowledgeRules: []
  }
}
function normalizedAssetDraft(asset = {}, relationships = []) {
  return {
    description: String(asset.description || ''),
    semanticColumns: cloneList(asset.semanticColumns),
    metrics: cloneList(asset.metrics),
    relationships: cloneList(relationships),
    terms: cloneList(asset.terms),
    knowledgeRules: cloneList(asset.knowledgeRules)
  }
}
function cloneValue(value) { return JSON.parse(JSON.stringify(value ?? null)) }
function cloneList(value) { return Array.isArray(value) ? cloneValue(value) : [] }
function sectionItems(section) {
  const value = assetDraft.value?.[section]
  return Array.isArray(value) ? value : []
}
function sectionLabel(section) {
  return semanticSectionMeta.find(item => item.id === section)?.label || section
}
function selectSemanticSection(section) {
  semanticSection.value = section
  selectedSemanticIndex.value = 0
  semanticSearch.value = ''
}
function semanticItemIdentity(section, item, index = 0) {
  const key = {
    semanticColumns: item?.columnName,
    metrics: item?.metricKey || item?.key,
    relationships: item?.name,
    terms: item?.term,
    knowledgeRules: item?.ruleId || item?.title
  }[section]
  return String(key || `未命名-${index + 1}`)
}
function semanticItemTitle(item, index = 0) {
  return String(
    item?.businessName
    || item?.term
    || item?.title
    || item?.name
    || item?.columnName
    || item?.metricKey
    || `未命名${activeSectionLabel.value}${index + 1}`
  )
}
function semanticItemSubtitle(item) {
  if (semanticSection.value === 'semanticColumns') return item.columnName || item.role || '待填写字段名'
  if (semanticSection.value === 'metrics') return item.metricKey || item.formula || '待填写指标编码'
  if (semanticSection.value === 'relationships') return [item.leftTable, item.rightTable].filter(Boolean).join(' → ') || '待选择关联表'
  if (semanticSection.value === 'terms') return listText(item.relatedColumns) || '待关联字段'
  return item.alwaysApply ? '强制规则' : '条件规则'
}
function newSemanticItem(section) {
  if (section === 'semanticColumns') return { columnName: '', businessName: '', role: 'DIMENSION', description: '', aliases: [], comparisonPolicy: 'exact', canonicalEntityRef: '', filterOperators: ['EQ', 'IN'] }
  if (section === 'metrics') return { metricKey: '', businessName: '', formula: '', unit: '', description: '', sourceColumns: [], aliases: [], metricLevel: 'business', metricGrain: '', aggregationPolicy: 'period_rollup', timeColumn: '' }
  if (section === 'relationships') return { name: '', leftTable: assetTable.value, rightTable: assetTables.value.find(name => name !== assetTable.value) || '', joinType: 'LEFT', keys: [['', '']], grain: '', cardinality: 'one_to_many', fanoutPolicy: 'DIRECTIONAL_GRAIN_GUARD', dedupKeys: [], rowIdentityPreserved: { leftToRight: false, rightToLeft: true }, useCases: [], cautions: [] }
  if (section === 'terms') return { term: '', description: '', aliases: [], relatedColumns: [] }
  return { title: '', content: '', alwaysApply: false, keywords: [] }
}
function addSemanticItem() {
  const items = sectionItems(semanticSection.value)
  items.push(newSemanticItem(semanticSection.value))
  selectedSemanticIndex.value = items.length - 1
  semanticSearch.value = ''
}
function removeSemanticItem() {
  const items = sectionItems(semanticSection.value)
  if (!items.length) return
  items.splice(selectedSemanticIndex.value, 1)
  selectedSemanticIndex.value = Math.max(0, Math.min(selectedSemanticIndex.value, items.length - 1))
}
function listText(value, separator = ', ') {
  return Array.isArray(value) ? value.filter(Boolean).join(separator) : ''
}
function updateListField(item, field, event) {
  const value = String(event?.target?.value || '')
  item[field] = value
    .replaceAll('，', ',')
    .replaceAll('\n', ',')
    .split(',')
    .map(part => part.trim())
    .filter(Boolean)
}
function addRelationshipKey() {
  if (!Array.isArray(selectedSemanticItem.value.keys)) selectedSemanticItem.value.keys = []
  selectedSemanticItem.value.keys.push(['', ''])
}
function updateRelationshipKey(pairIndex, valueIndex, event) {
  if (!Array.isArray(selectedSemanticItem.value.keys?.[pairIndex])) return
  selectedSemanticItem.value.keys[pairIndex][valueIndex] = String(event?.target?.value || '')
}
function removeRelationshipKey(pairIndex) {
  selectedSemanticItem.value.keys.splice(pairIndex, 1)
}
function stableComparable(value) {
  if (Array.isArray(value)) return value.map(stableComparable)
  if (value && typeof value === 'object') {
    return Object.keys(value).sort().reduce((result, key) => {
      if (!['evidence', 'confidence', 'updatedAt', 'reviewedAt', 'reviewer', 'reviewNote'].includes(key)) {
        result[key] = stableComparable(value[key])
      }
      return result
    }, {})
  }
  return value ?? ''
}
function equalSemanticValue(left, right) {
  return JSON.stringify(stableComparable(left)) === JSON.stringify(stableComparable(right))
}
function activeSectionItems(section) {
  const value = activeAssetSnapshot.value?.[section]
  return Array.isArray(value) ? value : []
}
function itemDiffStatus(section, item) {
  const identity = semanticItemIdentity(section, item)
  const active = activeSectionItems(section).find((candidate, index) => semanticItemIdentity(section, candidate, index) === identity)
  if (!active) return 'added'
  return equalSemanticValue(active, item) ? 'unchanged' : 'changed'
}
function changedFields(before, after) {
  const ignored = new Set(['evidence', 'confidence', 'updatedAt', 'reviewedAt', 'reviewer', 'reviewNote'])
  const keys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})])
  return [...keys].filter(key => !ignored.has(key) && !equalSemanticValue(before?.[key], after?.[key]))
}
function buildDiffEntries() {
  const result = []
  if (activeAssetSnapshot.value.description !== assetDraft.value.description) {
    result.push({
      section: 'description',
      identity: 'description',
      title: '表业务说明',
      status: 'changed',
      fields: ['description'],
      before: { description: activeAssetSnapshot.value.description },
      after: { description: assetDraft.value.description }
    })
  }
  for (const section of ['semanticColumns', 'metrics', 'relationships', 'terms', 'knowledgeRules']) {
    const active = activeSectionItems(section)
    const draft = sectionItems(section)
    const activeById = new Map(active.map((item, index) => [semanticItemIdentity(section, item, index), item]))
    const draftById = new Map(draft.map((item, index) => [semanticItemIdentity(section, item, index), item]))
    for (const [identity, item] of draftById) {
      const before = activeById.get(identity)
      if (!before) {
        result.push({ section, identity, title: semanticItemTitle(item), status: 'added', fields: Object.keys(item), before: null, after: item })
      } else if (!equalSemanticValue(before, item)) {
        result.push({ section, identity, title: semanticItemTitle(item), status: 'changed', fields: changedFields(before, item), before, after: item })
      }
    }
    for (const [identity, item] of activeById) {
      if (!draftById.has(identity)) {
        result.push({ section, identity, title: semanticItemTitle(item), status: 'removed', fields: [], before: item, after: null })
      }
    }
  }
  return result
}
function diffStatusLabel(status) {
  return { added: '新增', changed: '修改', removed: '删除', unchanged: '未变化' }[status] || status
}
function diffSummary(entry) {
  if (entry.status === 'added') return '将新增到正式语义层'
  if (entry.status === 'removed') return '发布后将从正式语义层移除'
  if (entry.section === 'description') return '表业务说明已调整'
  const fields = entry.fields.slice(0, 4).map(semanticFieldLabel)
  const suffix = entry.fields.length > fields.length ? ` 等 ${entry.fields.length} 个字段` : ''
  return `变更：${fields.join('、')}${suffix}`
}
function semanticFieldLabel(field) {
  return {
    description: '业务说明',
    columnName: '物理字段名',
    businessName: '业务名称',
    role: '字段角色',
    comparisonPolicy: '比较策略',
    aliases: '别名',
    canonicalEntityRef: '标准实体',
    filterOperators: '筛选符',
    metricKey: '指标编码',
    formula: '计算公式',
    unit: '单位',
    sourceColumns: '来源字段',
    metricLevel: '指标层级',
    metricGrain: '统计粒度',
    aggregationPolicy: '聚合策略',
    timeColumn: '时间字段',
    name: '关系编码',
    leftTable: '左表',
    rightTable: '右表',
    joinType: '连接类型',
    keys: '关联键',
    grain: '关系粒度',
    cardinality: '基数关系',
    fanoutPolicy: '防膨胀策略',
    term: '业务术语',
    relatedColumns: '关联字段',
    title: '规则标题',
    content: '规则内容',
    alwaysApply: '强制应用',
    keywords: '触发关键词'
  }[field] || field
}
function formatDiffValue(value) {
  if (value === undefined || value === null || value === '') return '未填写'
  if (Array.isArray(value)) {
    if (!value.length) return '无'
    return value.map(item => Array.isArray(item) ? item.join(' ↔ ') : String(item)).join('、')
  }
  if (typeof value === 'object') return Object.entries(value).map(([key, item]) => `${key}: ${item}`).join('、')
  if (typeof value === 'boolean') return value ? '是' : '否'
  return String(value)
}
function conflictReport(item) { return item?.payload?.conflictCheck || item?.conflictCheck || null }
function applyConflictResult(item, result) {
  const report = result.conflictCheck || result.suggestion?.payload?.conflictCheck
  if (!report) return
  item.payload = { ...(item.payload || {}), conflictCheck: report }
  item.conflictReviewStatus = result.suggestion?.conflictReviewStatus || 'required'
  item.conflictError = ''
  if (!(report.resolutionOptions || []).includes(item.conflictResolution)) item.conflictResolution = ''
}
function resolutionReady(item) {
  if (!item.conflictResolution || !(conflictReport(item)?.resolutionOptions || []).includes(item.conflictResolution)) return false
  return item.conflictResolution !== 'merge' || Boolean(String(item.mergedContent || '').trim())
}
function publishable(item) {
  const conflictReady = String(item.conflictReviewStatus || '').toLowerCase() !== 'required'
  return conflictReady && ['approved', 'publish_requested', 'published', 'indexed'].includes(String(item.status || '').toLowerCase()) && item.topic && item.sourceTable
}
function relationLabel(value) {
  return {
    duplicate: '重复',
    supplement: '可补充',
    conflict: '冲突',
    possible_conflict: '疑似冲突',
    different_scope: '适用范围不同'
  }[value] || value || '待判断'
}
function resolutionLabel(value) {
  return {
    use_existing: '沿用已有知识',
    replace: '用新知识替换旧知识',
    merge: '融合为一条知识',
    keep_both: '按不同范围分别保留',
    cancel: '取消本次入库'
  }[value] || value
}
function similarityPercent(value) { return `${Math.round(Number(value || 0) * 100)}%` }
function shortReportId(value) { return String(value || '').slice(0, 12) || '待生成' }
function statusLabel(value) {
  return {
    STALE_CONFLICT_REPORT: '冲突报告已变化，请根据最新结果重新确认',
    MERGED_CONTENT_REQUIRED: '请填写融合后的正式知识内容',
    MANUAL_CONFLICT_RESOLUTION_REQUIRED: '跨资产或跨类型冲突需要手工编辑语义资产',
    PREFLIGHT_FAILED: '预检未通过，请先修正语义资产',
    SELF_REVIEW_FORBIDDEN: '提交人不能审核自己的修改',
    SELF_PUBLISH_FORBIDDEN: '提交人不能发布自己的修改',
    DRAFT_CHANGED_AFTER_SUBMISSION: '草稿提交后已发生变化，请重新提交',
    APPROVAL_STALE: '审核通过后草稿已变化，请重新审核',
    REVIEW_REQUIRED: '必须先完成审核才能发布',
    INVALID_REVIEW_STATE: '当前状态不能执行审核'
  }[value] || value || '待处理'
}
</script>

<style scoped>
.governance-backdrop{position:fixed;inset:0;z-index:80;background:rgba(15,23,42,.42);display:grid;place-items:center;padding:24px}.governance-console{width:min(1180px,96vw);height:min(840px,94vh);overflow:hidden;background:#f8fbff;border-radius:22px;box-shadow:0 28px 80px rgba(15,23,42,.28);display:grid;grid-template-rows:auto auto 1fr}.governance-console>header{display:flex;justify-content:space-between;padding:22px 24px 16px;background:#fff}.governance-console h2,.governance-console h3,.governance-console p{margin:0}.governance-console header span{color:#4779d6;font-size:12px;font-weight:800}.governance-console header p,.section-title p{margin-top:5px;color:#778195;font-size:13px}.close{background:#eef3fa;border-radius:10px;width:38px;height:38px}.governance-console nav{display:flex;gap:8px;padding:10px 24px;background:#fff;border-top:1px solid #eef2f7;border-bottom:1px solid #e5ebf4}.governance-console nav button{display:flex;gap:7px;align-items:center;padding:9px 14px;border-radius:10px;background:transparent;color:#64748b}.governance-console nav button.active{background:#eaf2ff;color:#2563eb;font-weight:800}.console-content{overflow:auto;padding:22px 24px}.section-title{display:flex;justify-content:space-between;margin-bottom:16px}.section-title button,.governance-item button,.plan-card button{padding:8px 12px;border-radius:9px}.governance-item{background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px;margin-bottom:12px}.governance-item>div{display:flex;justify-content:space-between}.governance-item span{color:#2563eb;font-size:12px}.governance-item p{margin:10px 0;color:#475569}.governance-item footer{display:flex;justify-content:flex-end;gap:8px;margin-top:14px}.publish-targets{display:grid!important;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px}.publish-targets input{min-width:0;border:1px solid #dbe3ef;border-radius:9px;padding:9px;background:#fbfdff}.primary{background:#2563eb!important;color:white}.secondary{background:#eaf2ff!important;color:#2563eb}.builder-form{display:grid;grid-template-columns:1fr 1fr;gap:14px;background:#fff;padding:18px;border-radius:14px}.builder-form label{display:grid;gap:7px;font-size:13px;font-weight:700}.builder-form .wide{grid-column:1/-1}.builder-form input,.builder-form select,.builder-form textarea{border:1px solid #dbe3ef;border-radius:9px;padding:10px;background:#fff}.builder-form button{width:max-content;padding:10px 16px;border-radius:10px}.plan-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.plan-card{display:grid;gap:9px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:16px}.plan-card span{font-size:12px;color:#64748b}.console-state{display:flex;align-items:center;justify-content:center;gap:8px}.error{color:#b91c1c}.empty{padding:40px;text-align:center;color:#94a3b8}pre{white-space:pre-wrap;background:#0f172a;color:#dbeafe;padding:14px;border-radius:12px;max-height:260px;overflow:auto}@media(max-width:760px){.plan-grid{grid-template-columns:1fr}.builder-form{grid-template-columns:1fr}.governance-backdrop{padding:0}.governance-console{width:100vw;height:100vh;border-radius:0}}
.asset-editor-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:26px 0 12px}.asset-editor-head p{color:#778195;font-size:13px;margin-top:5px}.asset-picker{display:flex;gap:8px}.asset-picker select{max-width:210px;border:1px solid #dbe3ef;border-radius:9px;padding:9px;background:#fff}.asset-workbench{display:grid;gap:12px}.review-workflow-card{display:grid;gap:13px;padding:15px;border:1px solid #dbe5f2;border-radius:14px;background:#fff}.review-workflow-head{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.review-workflow-head>div{display:flex;align-items:center;gap:9px;margin-right:auto}.review-workflow-head span,.review-workflow-head small{color:#64748b;font-size:12px}.review-workflow-head b{color:#1d4ed8}.review-steps{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.review-steps span{padding:8px 9px;border-radius:8px;background:#f1f5f9;color:#94a3b8;font-size:12px;text-align:center}.review-steps span.active{background:#e8f1ff;color:#2563eb;font-weight:800}.review-workflow-card label{display:grid;gap:6px;color:#475569;font-size:12px;font-weight:700}.review-workflow-card input{border:1px solid #dbe3ef;border-radius:9px;padding:9px;background:#fbfdff}.asset-actions{display:flex;justify-content:flex-end;gap:8px;flex-wrap:wrap}.asset-actions button,.history-list button{padding:8px 12px;border-radius:9px}.asset-actions .danger{background:#fff1f2;color:#be123c;border:1px solid #fecdd3}.governance-summary{display:grid;grid-template-columns:1fr 1fr;gap:12px}.governance-summary article{min-width:0;background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:15px}.governance-summary article>span{float:right;color:#64748b;font-size:12px}.governance-summary article p{clear:both;padding-top:12px;color:#64748b;font-size:13px;line-height:1.55}.history-list{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:14px}.history-list article{display:grid;grid-template-columns:100px 1fr 180px auto;align-items:center;gap:10px;padding:9px 0;border-top:1px solid #edf2f7}.history-list span,.history-list small{color:#64748b;font-size:12px}@media(max-width:760px){.asset-editor-head{align-items:stretch;flex-direction:column}.asset-picker{flex-direction:column}.review-steps{grid-template-columns:1fr 1fr}.governance-summary{grid-template-columns:1fr}.history-list article{grid-template-columns:1fr 1fr}}
.semantic-editor{overflow:hidden;border:1px solid #dbe5f2;border-radius:15px;background:#fff}.semantic-editor-summary{display:flex;align-items:center;justify-content:space-between;gap:18px;padding:17px 18px;border-bottom:1px solid #e7edf5;background:linear-gradient(135deg,#f8fbff,#f2f7ff)}.semantic-editor-summary>div:first-child>span{color:#2563eb;font-size:11px;font-weight:800;letter-spacing:.08em}.semantic-editor-summary h4{margin:3px 0 2px;color:#172033;font-size:18px}.semantic-editor-summary p{color:#64748b;font-size:12px}.change-totals{display:flex;gap:7px;flex-wrap:wrap;justify-content:flex-end}.change-total{padding:6px 9px;border-radius:999px;font-size:11px;font-weight:800}.change-total.added{background:#ecfdf5;color:#047857}.change-total.changed{background:#eff6ff;color:#1d4ed8}.change-total.removed{background:#fff1f2;color:#be123c}.semantic-section-nav{display:grid!important;grid-template-columns:repeat(6,minmax(0,1fr));gap:0!important;padding:0!important;border:0!important;border-bottom:1px solid #e7edf5!important;background:#fff!important}.semantic-section-nav button{justify-content:center!important;border-radius:0!important;padding:12px 8px!important;border-right:1px solid #edf2f7;background:#fff!important;color:#64748b!important}.semantic-section-nav button:focus-visible{outline:2px solid #2563eb!important;outline-offset:-2px}.semantic-section-nav button:last-child{border-right:0}.semantic-section-nav button.active{box-shadow:inset 0 -2px #2563eb;background:#f8fbff!important;color:#1d4ed8!important}.semantic-section-nav button b{margin-left:auto;padding:2px 6px;border-radius:999px;background:#edf2f7;color:#64748b;font-size:10px}.semantic-section-nav button.active b{background:#dbeafe;color:#1d4ed8}.semantic-editor-body{display:grid;grid-template-columns:270px minmax(0,1fr);min-height:430px}.semantic-item-list{border-right:1px solid #e7edf5;background:#f8fafc}.semantic-list-tools{display:flex;gap:7px;padding:12px;border-bottom:1px solid #e7edf5}.semantic-list-tools label{display:flex;align-items:center;gap:7px;min-width:0;flex:1;padding:0 9px;border:1px solid #dce5f0;border-radius:9px;background:#fff;color:#94a3b8}.semantic-list-tools input{min-width:0;width:100%;border:0;outline:0;padding:9px 0;background:transparent}.icon-button{display:grid;place-items:center;width:38px;border-radius:9px!important;background:#2563eb!important;color:#fff!important}.semantic-list-empty{display:grid;gap:10px;place-items:center;padding:70px 20px;color:#94a3b8;font-size:12px}.semantic-list-empty button{color:#2563eb;background:transparent}.semantic-list-item{position:relative;display:grid!important;width:100%;gap:3px!important;padding:12px 54px 12px 14px!important;border-bottom:1px solid #edf2f7;border-radius:0!important;background:transparent!important;text-align:left}.semantic-list-item:hover,.semantic-list-item.active{background:#fff!important}.semantic-list-item.active{box-shadow:inset 3px 0 #2563eb}.semantic-list-item span{overflow:hidden;color:#334155;font-size:13px;font-weight:800;text-overflow:ellipsis;white-space:nowrap}.semantic-list-item small{overflow:hidden;color:#94a3b8;font-size:11px;text-overflow:ellipsis;white-space:nowrap}.semantic-list-item i{position:absolute;right:10px;top:18px;padding:3px 6px;border-radius:999px;font-size:9px;font-style:normal;font-weight:800}.semantic-list-item i.added,.semantic-list-item i.unchanged{background:#ecfdf5;color:#047857}.semantic-list-item i.changed{background:#eff6ff;color:#1d4ed8}.semantic-list-item i.removed{background:#fff1f2;color:#be123c}.semantic-form{min-width:0;padding:18px}.semantic-form>header{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #edf2f7}.semantic-form>header>div{display:grid;gap:3px}.semantic-form>header span{color:#64748b!important;font-size:11px!important}.semantic-form>header b{color:#1e293b}.delete-button{display:flex;align-items:center;gap:5px;padding:7px 10px!important;border-radius:8px!important;background:#fff1f2!important;color:#be123c!important}.semantic-form-empty{display:grid;place-items:center;color:#94a3b8}.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.form-grid label{display:grid;align-content:start;gap:6px;color:#475569;font-size:12px;font-weight:700}.form-grid label.wide,.form-grid .wide{grid-column:1/-1}.form-grid input,.form-grid select,.form-grid textarea,.description-editor textarea{box-sizing:border-box;width:100%;min-width:0;border:1px solid #dbe3ef;border-radius:9px;padding:10px;background:#fbfdff;color:#1e293b;outline:0}.form-grid input:focus,.form-grid select:focus,.form-grid textarea:focus,.description-editor textarea:focus{border-color:#77a5ed;box-shadow:0 0 0 3px rgba(37,99,235,.08);background:#fff}.formula-field textarea{font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,monospace}.switch-label{display:flex!important;align-items:center!important;grid-auto-flow:column;justify-content:start;align-self:end;min-height:38px}.switch-label input{width:16px!important}.relationship-keys{display:grid;gap:8px}.relationship-keys>div:first-child{display:flex;align-items:center;justify-content:space-between}.relationship-keys button{display:flex;align-items:center;gap:4px;padding:5px 8px;border-radius:7px;color:#2563eb;background:#eff6ff}.key-pair{display:grid;grid-template-columns:1fr auto 1fr auto;align-items:center;gap:8px}.key-pair span{color:#94a3b8;font-size:11px}.key-pair button{display:grid;place-items:center;width:32px;height:36px;color:#be123c;background:#fff1f2}.description-editor{display:grid;grid-template-columns:minmax(0,2fr) minmax(220px,1fr);gap:18px;padding:20px}.description-editor label{display:grid;gap:7px;color:#475569;font-size:12px;font-weight:700}.description-editor aside{padding:14px;border-radius:11px;background:#f1f5f9}.description-editor aside p{margin-top:8px;color:#64748b;font-size:12px;line-height:1.6}.visual-diff{overflow:hidden;border:1px solid #dbe5f2;border-radius:14px;background:#fff}.visual-diff>header{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid #e7edf5}.visual-diff>header>div{display:flex;align-items:center;gap:9px;color:#2563eb}.visual-diff>header>div>div{display:grid;gap:2px}.visual-diff>header b{color:#1e293b}.visual-diff>header span{color:#64748b;font-size:11px}.diff-empty{padding:26px;text-align:center;color:#94a3b8;font-size:12px}.diff-list{max-height:320px;overflow:auto}.diff-list article{display:grid;grid-template-columns:54px 190px minmax(0,1fr);align-items:center;gap:12px;padding:11px 16px;border-top:1px solid #edf2f7}.diff-list article:first-child{border-top:0}.diff-list article>span{width:max-content;padding:4px 7px;border-radius:999px;font-size:10px;font-weight:800}.diff-list article>span.added{background:#ecfdf5;color:#047857}.diff-list article>span.changed{background:#eff6ff;color:#1d4ed8}.diff-list article>span.removed{background:#fff1f2;color:#be123c}.diff-list article>div{display:grid;gap:2px}.diff-list article small{color:#94a3b8;font-size:10px}.diff-list article p{overflow:hidden;color:#64748b;font-size:12px;text-overflow:ellipsis;white-space:nowrap}.diff-list .diff-values{grid-column:2/-1;display:grid;gap:6px}.diff-values>div{display:grid;grid-template-columns:100px minmax(0,1fr) auto minmax(0,1fr);align-items:center;gap:8px;padding:7px 9px;border-radius:8px;background:#f8fafc}.diff-values em{color:#64748b;font-size:10px;font-style:normal}.diff-values del,.diff-values ins{overflow:hidden;font-size:11px;text-decoration:none;text-overflow:ellipsis;white-space:nowrap}.diff-values del{color:#be123c}.diff-values ins{color:#047857}.diff-values span{color:#94a3b8!important}@media(max-width:900px){.semantic-section-nav{grid-template-columns:repeat(3,1fr)}.semantic-editor-body{grid-template-columns:220px minmax(0,1fr)}}@media(max-width:760px){.semantic-editor-summary{align-items:flex-start;flex-direction:column}.change-totals{justify-content:flex-start}.semantic-section-nav{grid-template-columns:repeat(2,1fr)}.semantic-editor-body{grid-template-columns:1fr}.semantic-item-list{max-height:220px;overflow:auto;border-right:0;border-bottom:1px solid #e7edf5}.form-grid,.description-editor{grid-template-columns:1fr}.form-grid label.wide,.form-grid .wide{grid-column:1}.diff-list article{grid-template-columns:48px 1fr}.diff-list article p,.diff-list .diff-values{grid-column:1/-1;white-space:normal}.diff-values>div{grid-template-columns:80px 1fr}.diff-values>div span{display:none}.diff-values del,.diff-values ins{white-space:normal}}
.conflict-review{margin-top:14px;padding:14px;border:1px solid #f5c96a;border-radius:12px;background:#fffbeb}.conflict-review>div:first-child{display:flex;align-items:center;justify-content:space-between;gap:12px}.conflict-review>div:first-child span{color:#9a6700}.conflict-review>p{color:#7c5b12}.conflict-review ul{display:grid;gap:8px;margin:10px 0;padding:0;list-style:none}.conflict-review li{display:grid;grid-template-columns:1fr auto;gap:4px 12px;padding:10px;border:1px solid #f6df9d;border-radius:9px;background:#fff}.conflict-review li p,.conflict-review li small{grid-column:1/-1;margin:0}.conflict-review li small{color:#8b6d2d}.conflict-resolution{display:grid!important;grid-template-columns:minmax(200px,1fr) auto;align-items:end;gap:10px}.conflict-resolution label{display:grid;gap:6px;color:#6b4f0c;font-size:12px;font-weight:700}.conflict-resolution .wide{grid-column:1/-1}.conflict-resolution select,.conflict-resolution textarea{box-sizing:border-box;width:100%;border:1px solid #e7c66f;border-radius:9px;padding:9px;background:#fff}.conflict-resolution button{height:38px}@media(max-width:760px){.conflict-resolution{grid-template-columns:1fr}.conflict-review li{grid-template-columns:1fr}}
.conflict-review .inline-error{padding:8px 10px;border-radius:8px;background:#fef2f2;color:#b91c1c}
</style>
