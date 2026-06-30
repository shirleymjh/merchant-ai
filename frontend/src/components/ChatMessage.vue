<template>
  <article :class="['message', role]">
    <div v-if="role === 'assistant'" class="assistant-message">
      <div class="assistant-avatar">E</div>
      <div class="assistant-stream">
        <div class="message-meta">
          <strong>Evan</strong>
          <span>· {{ displayTime }}</span>
        </div>
        <div class="assistant-card">
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
          <MetricLineChart
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
      <div v-if="toastMessage" class="app-toast">{{ toastMessage }}</div>
    </Teleport>
  </article>
</template>

<script setup>
import { computed, ref } from 'vue'
import { Check, Copy, Download, Filter, Maximize2, ThumbsDown, ThumbsUp, X } from 'lucide-vue-next'
import MetricLineChart from './MetricLineChart.vue'

const props = defineProps({
  id: String,
  role: {
    type: String,
    required: true
  },
  text: {
    type: String,
    required: true
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
  feedbackStatus: {
    type: Object,
    default: () => ({})
  }
})

const displayTime = new Date().toLocaleString('zh-CN', {
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit'
}).replace(/\//g, '/')

const toastMessage = ref('')
const expandedTable = ref(null)
const tableFilters = ref({})
let toastTimer = null

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
        mode: inferAggregateMode(section?.title || '', rows)
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
    mode: inferAggregateMode('', rows)
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
    const line = cleanMarkdown(rawLine)
    if (!line) {
      flush()
      continue
    }
    if (/^([一二三四五六七八九十]+、|\d+[.、]\s*)/.test(line)) {
      flush()
      current.title = line
      continue
    }
    const bulletMatch = line.match(/^[-*•]\s+(.+)$/)
    if (bulletMatch) {
      current.items.push(bulletMatch[1].trim())
      continue
    }
    if (!current.text) {
      current.text = line
    } else {
      current.text += `\n${line}`
    }
  }
  flush()
  return sections.length ? sections : [{ title: '', text: props.text, items: [] }]
})

function cleanMarkdown(rawLine) {
  return String(rawLine || '')
    .trim()
    .replace(/^#{1,6}\s*/, '')
    .replace(/^>\s*/, '')
    .replace(/\*\*/g, '')
    .replace(/`([^`]+)`/g, '$1')
    .trim()
}

function answerSelectedTitles(text) {
  const titles = new Set()
  const matcher = /【([^】]+)】/g
  let match = matcher.exec(String(text || ''))
  while (match) {
    titles.add(normalizeSectionTitle(match[1]))
    match = matcher.exec(String(text || ''))
  }
  return titles
}

function normalizeSectionTitle(title) {
  return String(title || '')
    .replace(/[／/]/g, '-')
    .replace(/\s+/g, '')
    .trim()
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
  const numeric = Number(String(value).replace(/,/g, ''))
  return Number.isFinite(numeric) ? numeric : null
}

function formatMetricSummaryValue(value, column) {
  const numeric = numericValue(value)
  if (numeric === null) return formatCell(value)
  if (/amt|amount|gmv|金额/i.test(String(column || ''))) {
    return `${formatCompactNumber(numeric)}元`
  }
  return formatCompactNumber(numeric)
}

function formatCompactNumber(value) {
  if (Math.abs(value) >= 10000) return `${(value / 10000).toFixed(2).replace(/\.00$/, '')}万`
  if (Number.isInteger(value)) return String(value)
  return value.toFixed(2).replace(/\.00$/, '')
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

const columnLabels = {
  group_value: '分组对象',
  metric_value: '指标值',
  sample_count: '记录数',
  pt: '日期',
  value: '数值',
  cnt: '数量',
  merchant_id: '商家编号',
  merchant_name: '商家名称',
  seller_id: '卖家编号',
  seller_name: '卖家名称',
  user_id: '用户编号',
  buyer_id: '买家编号',
  buyer_name: '买家昵称',
  order_id: '主订单号',
  sub_order_id: '订单号',
  sub_order_status_name: '订单状态',
  sku_name: '商品名称',
  sku_title: '商品标题',
  sku_cnt: '商品数量',
  sku_count: '商品数量',
  pay_amt: '退款金额',
  order_detail_cnt: '订单量',
  order_gmv_amt_1d: 'GMV',
  pay_gmv_amt_1d: '支付GMV',
  trade_success_gmv_amt_1d: '交易成功GMV',
  refund_amt_1d: '退款金额',
  seller_repay_amt_1d: '赔付金额',
  cs_ticket_cnt_1d: '咨询工单量',
  pay_order_cnt_1d: '支付订单量',
  pay_status_name: '支付状态',
  pay_way_name: '赔款方式',
  sub_order_create_time: '下单时间',
  refund_id: '退货单号',
  refund_status_name: '退款状态',
  refund_reason: '退款原因',
  refund_create_time: '退款时间',
  ticket_id: '工单编号',
  ticket_title: '工单标题',
  ticket_status_name: '工单状态',
  priority_name: '优先级',
  is_reopen: '是否二开',
  is_reminder: '是否催单',
  ticket_score: '工单评分',
  ticket_create_time: '工单时间',
  bill_id: '赔付单号',
  repay_amt: '赔付金额',
  repay_status_name: '赔付状态',
  create_time: '创建时间',
  modify_time: '变更时间',
  coupon_id: '券编号',
  template_title: '券模板标题',
  coupon_amt: '优惠金额',
  coupon_send_status_name: '发券状态',
  coupon_create_time: '发券时间',
  spu_id: '商品编号',
  spu_name: '商品名称',
  spu_status_name: '商品状态',
  audit_operate_type_name: '审核操作',
  is_audit_pass: '审核通过',
  audit_remark: '审核备注',
  spu_apply_create_time: '申请时间',
  appeal_id: '申诉编号',
  appeal_status_name: '申诉状态',
  apply_type_name: '申诉类型',
  reason: '申诉原因',
  deposit_recharge_id: '充值单号',
  trans_id: '交易流水号',
  currency: '币种',
  deposit_recharge_amt: '充值金额',
  remark: '备注',
  inbound_id: '入库单号',
  inbound_status_name: '入库状态',
  sku_id: '规格编号',
  inbound_cnt: '入库数量',
  warehouse_id: '仓库编号',
  check_status_name: '质检状态',
  identify_result_name: '鉴定结果',
  outbound_id: '出库单号',
  address_json: '地址信息',
  address_province_name: '省份',
  address_city_name: '城市',
  address_district_name: '区县',
  address_street_name: '街道',
  discount_amt: '优惠金额',
  discount_id: '优惠编号',
  discount_type_name: '优惠类型',
  freight_amt: '运费金额',
  logistic_id: '物流编号',
  express_id: '运单号',
  express_status_name: '物流状态',
  refund_type_name: '退款类型',
  refund_desc: '退款描述',
  responsible_party_name: '责任方',
  buyer_mobile: '买家手机号',
  refund_discount_amt: '退款优惠金额',
  company_name: '公司名称',
  merchant_type_name: '商户类型',
  brand_type_name: '资质类型',
  balance_type_name: '结算类型',
  mobile: '商家手机号',
  license_id: '营业执照号',
  contact_name: '联系人',
  business_address: '经营地址',
  send_address: '发货地址',
  refnd_address: '退货地址',
  bank_name: '开户行',
  bank_account: '银行账号',
  account_type_name: '账户类型',
  ship_model_name: '发货模式',
  is_invoice: '是否开票',
  is_unconditional_refund: '七天无理由',
  init_deposit_amt: '初始保证金',
  deposit_freeze: '冻结保证金',
  deposit_amt: '保证金',
  min_poundage: '手续费下限',
  max_poundage: '手续费上限',
  poundage_discount: '费率折扣'
}

const columnTokenLabels = {
  account: '账户',
  address: '地址',
  amt: '金额',
  appeal: '申诉',
  apply: '申请',
  audit: '审核',
  balance: '结算',
  bank: '银行',
  bill: '账单',
  biz: '业务',
  brand: '资质',
  buyer: '买家',
  category: '类目',
  check: '质检',
  city: '城市',
  close: '关闭',
  code: '编码',
  company: '公司',
  contact: '联系人',
  cnt: '数量',
  color: '颜色',
  coupon: '优惠券',
  create: '创建',
  currency: '币种',
  deposit: '保证金',
  desc: '描述',
  detail: '明细',
  discount: '优惠',
  district: '区县',
  express: '快递',
  fee: '费用',
  freight: '运费',
  freeze: '冻结',
  gmv: 'GMV',
  goods: '商品',
  id: '编号',
  identify: '鉴定',
  inbound: '入库',
  invoice: '发票',
  is: '是否',
  item: '商品',
  level1: '一级',
  level2: '二级',
  level3: '三级',
  license: '营业执照',
  logistic: '物流',
  merchant: '商家',
  mobile: '手机号',
  modify: '变更',
  name: '名称',
  note: '备注',
  order: '订单',
  outbound: '出库',
  party: '方',
  pay: '支付',
  poundage: '手续费',
  price: '价格',
  priority: '优先级',
  product: '商品',
  province: '省份',
  reason: '原因',
  recharge: '充值',
  refund: '退款',
  remark: '备注',
  repay: '赔付',
  responsible: '责任',
  seller: '卖家',
  send: '发货',
  sku: 'SKU',
  spu: '商品',
  status: '状态',
  street: '街道',
  sub: '子',
  subsidy: '补贴',
  ticket: '工单',
  time: '时间',
  title: '标题',
  trans: '交易',
  type: '类型',
  user: '用户',
  warehouse: '仓库',
  way: '方式'
}

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

function columnLabel(column) {
  const rawColumn = String(column)
  const mapped = columnLabels[rawColumn] || columnLabels[rawColumn.toLowerCase()]
  const label = mapped || humanizeColumnName(rawColumn)
  return shortenLabel(cleanLabel(label))
}

function cleanLabel(label) {
  return label
    .replace(/\s*\d+\s*[-、.:：].*$/g, '')
    .replace(/\s*包括[:：].*$/g, '')
    .replace(/_id$/i, '编号')
    .replace(/_/g, ' ')
    .replace(/[（(][^）)]*[）)]/g, '')
    .replace(/\b\d+\s*[-、.:：]?\s*[^,，;；\s]+/g, '')
    .replace(/[，,；;、]\s*\d+.*$/g, '')
    .replace(/\s+/g, '')
    .trim()
}

function shortenLabel(label) {
  if (label.length <= COLUMN_LABEL_LIMIT) return label
  return label
    .replace('营业执照', '执照')
    .replace('交易流水', '流水')
    .replace('编号', '号')
    .replace('金额元', '金额')
    .slice(0, COLUMN_LABEL_LIMIT)
}

function humanizeColumnName(column) {
  const tokens = column
    .replace(/([a-z])([A-Z])/g, '$1_$2')
    .toLowerCase()
    .split(/[_\s]+/)
    .filter(Boolean)
  const label = tokens.map((token) => columnTokenLabels[token] || token).join('')
  return label || column
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function inferAggregateMode(title, rows) {
  const text = `${title || ''} ${(rows || []).map(row => row?.group_value || '').join(' ')}`
  return /top|排行|最高|最低|前\d+/i.test(text) ? 'topn' : 'group'
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
  return /^[a-z][a-z0-9_]*$/i.test(String(title || ''))
}

function resolveGroupLabel(title) {
  const text = String(title || '')
  if (text.includes('商品')) return '商品'
  if (text.includes('原因')) return '原因'
  if (text.includes('状态')) return '状态'
  if (text.includes('工单')) return '对象'
  return '分组对象'
}

function resolveMetricLabel(title) {
  const text = String(title || '')
  if (text.includes('GMV') || text.includes('金额') || text.includes('销售额') || text.includes('成交额')) {
    return '金额'
  }
  if (text.includes('订单量') || text.includes('退款量') || text.includes('工单量') || text.includes('履约量')) {
    return '数量'
  }
  return '指标值'
}

function formatAggregateGroup(row) {
  return formatCell(row?.group_value)
}

function formatAggregateMetric(value, title) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return formatCell(value)
  }
  if (resolveMetricLabel(title) === '金额') {
    return `${numeric.toFixed(2)}元`
  }
  return Number.isInteger(numeric) ? `${numeric}` : numeric.toFixed(2)
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
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

function textEscape(value) {
  return String(value ?? '').replace(/\t/g, ' ').replace(/\n/g, ' ')
}

function safeFileName(name) {
  return String(name || '查询结果')
    .replace(/[\\/:*?"<>|]/g, '_')
    .replace(/\s+/g, '_')
    .slice(0, 60)
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

defineEmits(['feedback'])
</script>
