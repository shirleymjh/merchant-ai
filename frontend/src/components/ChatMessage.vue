<template>
  <article :class="['message', role]">
    <div v-if="role === 'assistant'" class="assistant-card">
      <div v-if="steps?.length" class="thinking">
        <div class="thinking-title">
          <Sparkles :size="16" />
          <span>思考完成</span>
        </div>
        <div v-for="step in steps" :key="step" class="thinking-step">
          <CircleCheck :size="15" />
          <span>{{ step }}</span>
        </div>
      </div>
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
      <div v-if="dataRows?.length" class="detail-table-wrap">
        <table class="detail-table">
          <thead>
            <tr>
              <th v-for="column in tableColumns" :key="column" :title="columnLabel(column)">{{ columnLabel(column) }}</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(row, rowIndex) in dataRows" :key="rowIndex">
              <td v-for="column in tableColumns" :key="column">{{ formatCell(row[column]) }}</td>
            </tr>
          </tbody>
        </table>
      </div>
      <div v-if="tables?.length" class="table-tags">
        <span v-for="table in tables" :key="table">{{ table }}</span>
      </div>
      <div v-if="id" class="message-actions">
        <button
          type="button"
          :class="['adopt-action', { active: feedbackStatus?.adopted }]"
          title="采纳"
          @click="$emit('feedback', { id, adopted: true })"
        >
          <Check :size="16" />
          <span>{{ feedbackStatus?.adopted ? '已采纳' : '采纳' }}</span>
        </button>
        <button
          type="button"
          :class="{ active: feedbackStatus?.liked }"
          title="点赞"
          @click="$emit('feedback', { id, liked: true, disliked: false })"
        >
          <ThumbsUp :size="16" />
        </button>
        <button
          type="button"
          :class="{ active: feedbackStatus?.disliked }"
          title="点踩"
          @click="$emit('feedback', { id, liked: false, disliked: true })"
        >
          <ThumbsDown :size="16" />
        </button>
      </div>
      <p class="ai-note">内容为 AI 生成，仅供参考</p>
    </div>
    <div v-else class="user-bubble">{{ text }}</div>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import { Check, CircleCheck, Sparkles, ThumbsDown, ThumbsUp } from 'lucide-vue-next'

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
  feedbackStatus: {
    type: Object,
    default: () => ({})
  }
})

const tableColumns = computed(() => {
  const columns = []
  for (const row of props.dataRows || []) {
    for (const column of Object.keys(row || {})) {
      if (!columns.includes(column)) {
        columns.push(column)
      }
    }
  }
  return columns
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
    const line = rawLine.trim()
    if (!line) {
      flush()
      continue
    }
    if (/^[一二三四五六七八九十]+、/.test(line)) {
      flush()
      current.title = line
      continue
    }
    if (line.startsWith('- ')) {
      current.items.push(line.slice(2).trim())
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

const COLUMN_LABEL_LIMIT = 10

const columnLabels = {
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
  pay_amt: '支付金额',
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

defineEmits(['feedback'])
</script>
