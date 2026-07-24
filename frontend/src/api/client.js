function configuredMerchantId() {
  return String(globalThis.__MERCHANT_AI_RUNTIME__?.merchantId || '').trim()
}

function configuredDevOpsActor() {
  const runtime = globalThis.__MERCHANT_AI_RUNTIME__ || {}
  if (runtime.internalMode !== true) return ''
  return String(
    runtime.opsActor
    || runtime.identity?.userId
    || runtime.identity?.displayName
    || ''
  ).trim()
}

function merchantIdFrom(context = {}, options = {}) {
  return String(
    options.merchantId
    || context?.userIdentity?.merchantId
    || context?.user_identity?.merchant_id
    || configuredMerchantId()
    || ''
  ).trim()
}

function withMerchantId(payload, merchantId) {
  const value = String(merchantId || '').trim()
  return value ? { ...payload, merchantId: value } : payload
}

function merchantQuery(merchantId) {
  const value = String(merchantId || '').trim()
  return value ? `?merchantId=${encodeURIComponent(value)}` : ''
}

async function request(path, options = {}) {
  const { headers: optionHeaders = {}, ...requestOptions } = options
  const response = await fetch(path, {
    credentials: 'same-origin',
    ...requestOptions,
    headers: {
      'Content-Type': 'application/json',
      ...optionHeaders
    }
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return response.json()
}

async function opsRequest(path, options = {}) {
  const actor = configuredDevOpsActor()
  return request(path, {
    ...options,
    headers: {
      ...(options.headers || {}),
      ...(actor ? { 'X-Dev-Ops-Actor': actor } : {})
    }
  })
}

export async function sendMessage(message, context, messageHistory = []) {
  return request('/api/chat', {
    method: 'POST',
    body: JSON.stringify(withMerchantId({ message, context, messageHistory }, merchantIdFrom(context)))
  })
}

export async function startAsyncRun(message, context, options = {}) {
  return request('/api/runs/async', {
    method: 'POST',
    body: JSON.stringify({
      message,
      ...withMerchantId({}, merchantIdFrom(context, options)),
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
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message,
      ...withMerchantId({}, merchantIdFrom(context, options)),
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

export async function uploadAttachment(file, signal, merchantId = '') {
  const params = new URLSearchParams({ name: file.name, type: file.type || 'application/octet-stream' })
  const effectiveMerchantId = String(merchantId || configuredMerchantId()).trim()
  if (effectiveMerchantId) params.set('merchantId', effectiveMerchantId)
  const response = await fetch(`/api/attachments?${params}`, {
    method: 'POST',
    headers: { 'Content-Type': file.type || 'application/octet-stream' },
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
      ...withMerchantId({}, merchantIdFrom(context, options)),
      threadId: options.threadId || '',
      context,
      messageHistory: options.messageHistory || [],
      attachments: options.attachments || [],
      userIdentity: options.userIdentity || {}
    }),
    signal: options.signal
  })
}

export async function getMerchantProfile(merchantId = '') {
  return request(`/api/merchant-profile${merchantQuery(merchantId || configuredMerchantId())}`)
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
    body: JSON.stringify(withMerchantId(payload, payload?.merchantId || configuredMerchantId()))
  })
}

export async function actOnKnowledgeSuggestion(id, action, payload = {}) {
  return request(`/api/merchant/knowledge-suggestions/${encodeURIComponent(id)}/action`, {
    method: 'POST',
    body: JSON.stringify({
      action,
      ...withMerchantId({}, payload.merchantId || configuredMerchantId()),
      actor: payload.actor || '',
      note: payload.note || '',
      conflictResolution: payload.conflictResolution || ''
    })
  })
}

export async function getDailyReport(merchantId = '') {
  return request(`/api/daily-report${merchantQuery(merchantId || configuredMerchantId())}`)
}

export async function getTopics() {
  return opsRequest('/api/topics')
}

export async function buildTopicAsset(payload) {
  return opsRequest('/api/topics/build', { method: 'POST', body: JSON.stringify(payload) })
}

export async function getTopicAssets(topic) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/assets`)
}

export async function getTopicTableGovernance(topic, tableName) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/governance`)
}

export async function saveTopicTableDraft(topic, tableName, payload) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/draft`, { method: 'POST', body: JSON.stringify(payload) })
}

export async function submitTopicTableReview(topic, tableName, payload = {}) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/submit-review`, {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export async function reviewTopicTable(topic, tableName, payload) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/review`, {
    method: 'POST',
    body: JSON.stringify(payload)
  })
}

export async function publishTopicTable(topic, tableName, payload) {
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/publish`, { method: 'POST', body: JSON.stringify(payload) })
}

export async function rollbackTopicTable(topic, tableName, version = '', payload = {}) {
  const params = new URLSearchParams({ version })
  if (payload.reviewer) params.set('reviewer', payload.reviewer)
  if (payload.reason) params.set('reason', payload.reason)
  return opsRequest(`/api/topics/${encodeURIComponent(topic)}/tables/${encodeURIComponent(tableName)}/rollback?${params}`, { method: 'POST' })
}

export async function getKnowledgeSuggestions(status = '', merchantId = '') {
  const params = new URLSearchParams()
  if (status) params.set('status', status)
  const effectiveMerchantId = String(merchantId || configuredMerchantId()).trim()
  if (effectiveMerchantId) params.set('merchantId', effectiveMerchantId)
  const suffix = params.toString() ? `?${params}` : ''
  return opsRequest(`/api/ops/knowledge-suggestions${suffix}`)
}

export async function checkKnowledgeSuggestionConflicts(id, merchantId = '') {
  return opsRequest(`/api/ops/knowledge-suggestions/${encodeURIComponent(id)}/conflict-check${merchantQuery(merchantId || configuredMerchantId())}`, {
    method: 'POST'
  })
}

export async function reviewKnowledgeSuggestion(id, payload) {
  return opsRequest(`/api/ops/knowledge-suggestions/${encodeURIComponent(id)}/review${merchantQuery(payload?.merchantId || configuredMerchantId())}`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}

export async function publishKnowledgeSuggestion(id, payload) {
  return opsRequest(`/api/ops/knowledge-suggestions/${encodeURIComponent(id)}/publish${merchantQuery(payload?.merchantId || configuredMerchantId())}`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}

export async function getAnalysisCatalog() {
  return opsRequest('/api/ops/skill-market')
}

export async function installAnalysisPlan(name, payload = {}) {
  return opsRequest(`/api/ops/skill-market/${encodeURIComponent(name)}/install`, {
    method: 'POST', body: JSON.stringify(payload)
  })
}
