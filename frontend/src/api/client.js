const DEFAULT_MERCHANT_ID = '100'
const OPS_TOKEN = import.meta.env.VITE_OPS_TOKEN || ''

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: {
      'Content-Type': 'application/json',
      ...(OPS_TOKEN ? { 'X-Ops-Token': OPS_TOKEN } : {}),
      ...(options.headers || {})
    },
    ...options
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

export async function sendMessage(message, context, messageHistory = []) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify({ message, merchantId: DEFAULT_MERCHANT_ID, context, messageHistory })
  })
}

export async function startAsyncRun(message, context, options = {}) {
  return request('/api/runs/async', {
    method: 'POST',
    body: JSON.stringify({
      message,
      merchantId: DEFAULT_MERCHANT_ID,
      threadId: options.threadId || '',
      context,
      messageHistory: options.messageHistory || [],
      attachments: options.attachments || [],
      userIdentity: options.userIdentity || {}
    }),
    signal: options.signal
  })
}

export async function streamChatRun(message, context, options = {}, onEvent = () => {}) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...(OPS_TOKEN ? { 'X-Ops-Token': OPS_TOKEN } : {}) },
    body: JSON.stringify({
      message,
      merchantId: DEFAULT_MERCHANT_ID,
      threadId: options.threadId || '',
      context,
      messageHistory: options.messageHistory || [],
      attachments: options.attachments || [],
      userIdentity: options.userIdentity || {}
    }),
    signal: options.signal
  })
  if (!response.ok || !response.body) throw new Error(`HTTP ${response.status}`)
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let completed = null
  while (true) {
    const { done, value } = await reader.read()
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() || ''
    for (const frame of frames) {
      const dataLine = frame.split('\n').find(line => line.startsWith('data:'))
      if (!dataLine) continue
      const event = JSON.parse(dataLine.slice(5).trim())
      await onEvent(event)
      if (event.event === 'done') completed = event
      if (event.event === 'error') throw new Error(event.message || 'STREAM_FAILED')
    }
    if (done) break
  }
  return completed
}

export async function uploadAttachment(file, signal) {
  const params = new URLSearchParams({ name: file.name, type: file.type || 'application/octet-stream', merchantId: DEFAULT_MERCHANT_ID })
  const response = await fetch(`/api/attachments?${params}`, {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream', ...(OPS_TOKEN ? { 'X-Ops-Token': OPS_TOKEN } : {}) },
    body: file,
    signal
  })
  if (!response.ok) throw new Error(`HTTP ${response.status}`)
  return response.json()
}

export async function resumeChatRun(message, context, options = {}) {
  return request('/api/chat/resume', {
    method: 'POST',
    body: JSON.stringify({
      message,
      merchantId: DEFAULT_MERCHANT_ID,
      threadId: options.threadId || '',
      context,
      messageHistory: options.messageHistory || [],
      attachments: options.attachments || [],
      userIdentity: options.userIdentity || {}
    }),
    signal: options.signal
  })
}

export async function getMerchantProfile() {
  return request(`/api/merchant-profile/${DEFAULT_MERCHANT_ID}`)
}

export async function getRun(threadId, runId) {
  return request(`/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}`)
}

export async function getRunEvents(threadId, runId) {
  return request(`/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/events`)
}

export async function cancelRun(threadId, runId) {
  return request(`/api/threads/${encodeURIComponent(threadId)}/runs/${encodeURIComponent(runId)}/cancel`, {
    method: 'POST'
  })
}

export async function sendFeedback(id, payload) {
  return request(`/api/answers/${id}/feedback`, {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export async function recordMetricDefinitionPreference(payload) {
  return request('/api/merchant-preferences/metric-definition', {
    method: 'POST',
    body: JSON.stringify({ merchantId: DEFAULT_MERCHANT_ID, ...payload })
  })
}

export async function getDailyReport() {
  return request(`/api/daily-report?merchantId=${DEFAULT_MERCHANT_ID}`)
}

export async function getTopics() {
  return request('/api/topics')
}

export async function buildTopicAsset(payload) {
  return request('/api/topics/build', { method: 'POST', body: JSON.stringify(payload) })
}

export async function getTopicAssets(topic) {
  return request(`/api/topics/${encodeURIComponent(topic)}/assets`)
}

export async function getTopicTableGovernance(topic, tableName) {
  return request(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/governance`)
}

export async function saveTopicTableDraft(topic, tableName, payload) {
  return request(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/draft`, { method: 'POST', body: JSON.stringify(payload) })
}

export async function publishTopicTable(topic, tableName, payload) {
  return request(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/publish`, { method: 'POST', body: JSON.stringify(payload) })
}

export async function rollbackTopicTable(topic, tableName, version = '') {
  const params = new URLSearchParams({ version, reviewer: 'merchant_ops', reason: 'console rollback' })
  return request(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/rollback?${params}`, { method: 'POST' })
}

export async function getKnowledgeSuggestions(status = '') {
  const suffix = status ? `?status=${encodeURIComponent(status)}&merchantId=${DEFAULT_MERCHANT_ID}` : `?merchantId=${DEFAULT_MERCHANT_ID}`
  return request(`/api/ops/knowledge-suggestions${suffix}`)
}

export async function reviewKnowledgeSuggestion(id, payload) {
  return request(`/api/ops/knowledge-suggestions/${encodeURIComponent(id)}/review?merchantId=${DEFAULT_MERCHANT_ID}`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}

export async function publishKnowledgeSuggestion(id, payload) {
  return request(`/api/ops/knowledge-suggestions/${encodeURIComponent(id)}/publish?merchantId=${DEFAULT_MERCHANT_ID}`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}

export async function getAnalysisCatalog() {
  return request('/api/ops/skill-market')
}

export async function installAnalysisPlan(name, payload = {}) {
  return request(`/api/ops/skill-market/${encodeURIComponent(name)}/install`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}

export function mockDailyReport() {
  return {
    merchantId: '100',
    merchantName: 'yshopping商家100',
    date: '2026-05-23',
    metrics: {
      昨日总gmv金额: 0,
      昨日下单用户量: 0,
      昨日总订单量: 0,
      昨日交易成功订单量: 0,
      昨日退货量: 0,
      昨日退款金额: 0
    },
    suggestions: [
      '关注订单、退款和客服工单是否同步波动。',
      '可把重点指标加入经营日报，持续跟踪异常变化。'
    ],
    anomalyAlerts: [],
    drillDownActions: [
      { label: '查看订单趋势', question: '最近7天订单量和GMV按日趋势如何？', actionType: 'follow_up_question' },
      { label: '查看退款商品', question: '昨日退款金额最高的商品有哪些？', actionType: 'follow_up_question' }
    ],
    traceability: {
      sourceSummary: '演示数据',
      timeRange: '昨日',
      sourceTables: ['ads_merchant_profile']
    }
  }
}

export function mockChat(message) {
  const isGreeting = /^(你好|您好|hi|hello|hey|在吗|嗨)/i.test(message.trim())
  return {
    id: `mock_${Date.now()}`,
    answer: isGreeting
      ? '您好，我是 yshopping 商家 AI 助手，有任何经营、订单、退货、客服、赔付、优惠券、商品或商家资料问题都可以问我。'
      : '我已收到您的问题。当前后端未连接时展示的是前端演示回复；启动 Python 后端后会读取 Doris 数据并写入 merchant_ai_answer。',
    categoryName: isGreeting ? '未知' : '商家其他信息',
    persisted: false,
    dorisTables: [],
    suggestions: ['最近7天店铺整体经营情况怎么样？', '退款金额最高的前5个商品有哪些？', '工单最多的问题类型有哪些？'],
    merchantExperience: {
      businessAdvice: ['启动后端后会基于真实查询结果生成经营建议。'],
      suggestedQuestions: ['最近7天店铺整体经营情况怎么样？', '最近7天订单量和退款金额有什么变化？'],
      anomalyAlerts: [],
      metricDisclosures: [],
      traceability: { sourceSummary: '前端演示回复', evidenceStatus: 'demo' },
      drillDownActions: [],
      reportSubscriptionHint: {},
      clarificationHints: []
    },
    thinkingSteps: ['问题分析完成', '回答整理完成'],
    dataRows: []
  }
}
