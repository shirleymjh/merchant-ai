<template>
  <main class="evan-app">
    <section class="workspace-main">
      <header class="topic-header">
        <div class="topic-tabs">
          <button
            v-for="session in sessions"
            :key="session.id"
            :class="['topic-tab', { active: session.id === activeSessionId }]"
            type="button"
            :title="session.title"
            @click="switchSession(session.id)"
          >
            <span>{{ session.title }}</span>
          </button>
          <button
            :class="['topic-tab', 'new-session', { confirmed: newSessionFlash }]"
            type="button"
            @click="resetChat"
          >
            <Plus :size="15" />
            <span>{{ newSessionFlash ? '已开启' : '新会话' }}</span>
          </button>
        </div>
        <div class="topic-status">
          <Zap :size="14" />
          <span>经营数据已连接</span>
        </div>
      </header>

      <section class="chat-canvas" ref="chatList">
        <ChatMessage
          v-for="message in messages"
          :key="message.localId"
          :id="message.id"
          :role="message.role"
          :text="message.text"
          :steps="message.steps"
          :tables="message.tables"
          :data-rows="message.dataRows"
          :data-sections="message.dataSections"
          :feedback-status="message.feedbackStatus"
          @feedback="handleFeedback"
        />
        <div v-if="loading" class="loading-card">
          <LoaderCircle :size="18" />
          <span>{{ runStatusText }}</span>
          <button type="button" class="stop-run-button" title="停止回答" :disabled="stopping" @click="stopCurrentRun">
            <CircleStop :size="15" />
            <span>{{ stopping ? '停止中' : '停止' }}</span>
          </button>
        </div>
      </section>

      <div class="composer-area">
        <SuggestionList
          :suggestions="suggestions"
          :compact="hasConversation"
          @select="sendSuggestion"
          @refresh="rotateSuggestions"
        />
        <form class="input-bar" @submit.prevent="submit">
          <input ref="inputRef" v-model.trim="input" type="text" placeholder="说出您的疑惑吧，yshopping 帮你解决" />
          <button v-if="!loading" type="submit" class="send-button" title="发送" :disabled="!input">
            <Send :size="20" />
          </button>
          <button v-else type="button" class="send-button stop-button" title="停止回答" :disabled="stopping" @click="stopCurrentRun">
            <CircleStop :size="20" />
          </button>
        </form>
      </div>
    </section>

    <aside class="insight-rail">
      <DailyReportCard v-if="dailyReport" :report="dailyReport" :compact="true" />
    </aside>
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { CircleStop, LoaderCircle, Plus, Send, Zap } from 'lucide-vue-next'
import ChatMessage from './components/ChatMessage.vue'
import DailyReportCard from './components/DailyReportCard.vue'
import SuggestionList from './components/SuggestionList.vue'
import { cancelRun, getDailyReport, getRun, getRunEvents, mockChat, mockDailyReport, sendFeedback, startAsyncRun } from './api/client'

const input = ref('')
const loading = ref(false)
const stopping = ref(false)
const runStatusText = ref('正在分析问题并读取经营数据')
const activeRun = ref(null)
let submitController = null
let pollTimer = null
let chatEpoch = 0
let newSessionTimer = null
const chatList = ref(null)
const inputRef = ref(null)
const dailyReport = ref(null)
const newSessionFlash = ref(false)
const defaultSuggestions = [
  '我想查看保证金',
  '最近7天咨询工单量',
  '我要货品上架，具体规则有吗？',
  '昨天退款金额是多少？',
  '直接退款量是多少？',
  '最近10天订单明细',
  '商品审核被拒怎么办？',
  '上周供应链履约量是多少',
  '所有申诉表明细给我看看',
  '最近7天退款金额和退款量是多少？',
  '最近10天退款最多的商品有哪些？',
  '最近7天按工单状态统计工单量',
  '最近30天 GMV 为什么下降？',
  '最近7天订单量和退款金额有什么变化？',
  '上个月退款原因排行前5是什么？',
  '最近10天商品审核拒绝明细',
  '最近7天优惠金额和 GMV 表现如何？',
  '最近30天保证金充值流水有没有异常？',
  '最近7天履约量和发货超时订单量',
  '最近10天赔付金额最高的单据',
  '最近7天商品审核通过量和拒绝量',
  '最近30天申诉次数和处罚次数',
  '最近7天退款明细按金额排序',
  '最近10天工单明细按状态汇总',
  '最近30天订单量、退货量和催单量',
  '最近7天店铺整体经营情况怎么样？',
  '商品审核拒绝最多的商品有哪些？',
  '退款金额最高的前5单给我看一下',
  '催单工单最近是否升高？'
]
const suggestions = ref(defaultSuggestions.slice(0, 3))
const suggestionPool = ref(defaultSuggestions.slice())
const suggestionCursor = ref(0)
const suggestionPageSize = 3
const conversationContext = ref(null)
const initialSession = createConversationSession('经营分析工作台', '您好，我是 yshopping 商家 AI 助手，有经营问题欢迎随时问我。')
const sessions = ref([initialSession])
const activeSessionId = ref(initialSession.id)
const messages = ref(cloneValue(initialSession.messages))

const hasConversation = computed(() => messages.value.some(message => message.role === 'user'))
onMounted(async () => {
  try {
    dailyReport.value = await getDailyReport()
  } catch {
    dailyReport.value = mockDailyReport()
  }
})

onBeforeUnmount(() => {
  clearRunPoll()
  clearNewSessionTimer()
})

async function submit() {
  if (!input.value || loading.value) return
  const message = input.value
  input.value = ''
  await ask(message)
}

async function sendSuggestion(question) {
  if (loading.value) return
  await ask(question)
}

async function ask(message) {
  const epoch = chatEpoch
  clearNewSessionTimer()
  newSessionFlash.value = false
  clearRunPoll()
  clearSubmitController()
  messages.value.push({
    localId: `u_${Date.now()}`,
    role: 'user',
    text: message
  })
  saveActiveSessionSnapshot()
  loading.value = true
  stopping.value = false
  runStatusText.value = '正在提交任务'
  await scrollBottom()
  submitController = new AbortController()
  try {
    const created = await startAsyncRun(message, conversationContext.value, { signal: submitController.signal })
    submitController = null
    if (epoch !== chatEpoch) return
    const runId = created.runId
    const threadId = created.threadId
    if (!runId || !threadId) {
      throw new Error('RUN_CREATE_FAILED')
    }
    activeRun.value = {
      runId,
      threadId,
      token: `run_${Date.now()}_${Math.random().toString(16).slice(2)}`
    }
    runStatusText.value = '任务已提交，正在排队'
    scheduleRunPoll(activeRun.value.token, 300)
  } catch (error) {
    if (epoch !== chatEpoch) return
    const aborted = error?.name === 'AbortError'
    clearRunPoll()
    clearSubmitController()
    loading.value = false
    stopping.value = false
    if (aborted) {
      appendAssistant(systemMessage('已停止本次回答。您可以修改问题后重新提问。'))
    } else {
      appendAssistant(mockChat(message))
    }
    await scrollBottom()
  }
}

function scheduleRunPoll(token, delay = 900) {
  clearRunPoll()
  pollTimer = window.setTimeout(() => {
    pollActiveRun(token)
  }, delay)
}

async function pollActiveRun(token) {
  const current = activeRun.value
  if (!current || current.token !== token) return
  try {
    const [runPayload, eventsPayload] = await Promise.allSettled([
      getRun(current.threadId, current.runId),
      getRunEvents(current.threadId, current.runId)
    ])
    if (!activeRun.value || activeRun.value.token !== token) return
    const run = runPayload.status === 'fulfilled' ? runPayload.value?.run : null
    if (eventsPayload.status === 'fulfilled') {
      runStatusText.value = latestRunStatusText(run, eventsPayload.value?.events || [])
    } else {
      runStatusText.value = latestRunStatusText(run, [])
    }
    const status = String(run?.status || '').toUpperCase()
    if (status === 'COMPLETED') {
      clearRunPoll()
      loading.value = false
      stopping.value = false
      activeRun.value = null
      if (run.answer) {
        appendAssistant(run.answer)
      } else {
        appendAssistant(systemMessage('任务已完成，但没有返回可展示的答案。'))
      }
      await scrollBottom()
      return
    }
    if (status === 'FAILED') {
      clearRunPoll()
      loading.value = false
      stopping.value = false
      activeRun.value = null
      appendAssistant(systemMessage(`本次回答失败：${run?.error || '后端执行异常'}`))
      await scrollBottom()
      return
    }
    if (status === 'CANCELED') {
      clearRunPoll()
      loading.value = false
      stopping.value = false
      activeRun.value = null
      await scrollBottom()
      return
    }
    scheduleRunPoll(token)
  } catch {
    if (!activeRun.value || activeRun.value.token !== token) return
    runStatusText.value = '正在等待后端返回状态'
    scheduleRunPoll(token, 1200)
  }
}

async function stopCurrentRun() {
  if (stopping.value) return
  const current = activeRun.value
  stopping.value = true
  runStatusText.value = '正在停止本次回答'
  clearRunPoll()
  if (!current && submitController) {
    submitController.abort()
    return
  }
  if (!current) {
    loading.value = false
    stopping.value = false
    return
  }
  const stoppedRun = current
  activeRun.value = null
  try {
    await cancelRun(stoppedRun.threadId, stoppedRun.runId)
  } catch {
    // 取消请求失败也要允许用户继续提问；后端旧结果不会再被当前 token 接收。
  } finally {
    loading.value = false
    stopping.value = false
    appendAssistant(systemMessage('已停止本次回答。您可以修改问题后重新提问。'))
    await scrollBottom()
  }
}

function clearRunPoll() {
  if (pollTimer) {
    window.clearTimeout(pollTimer)
    pollTimer = null
  }
}

function clearSubmitController() {
  submitController = null
}

function latestRunStatusText(run, events = []) {
  const status = String(run?.status || '').toUpperCase()
  if (status === 'QUEUED') return '任务已提交，正在排队'
  if (status === 'RUNNING') {
    const lastEvent = [...events].reverse().find(event => event?.node && event.node !== 'RUN_MANAGER')
    if (lastEvent?.node) {
      return friendlyNodeStatus(lastEvent.node)
    }
    return '正在分析问题并读取经营数据'
  }
  if (status === 'FAILED') return '执行失败'
  if (status === 'CANCELED') return '已停止'
  return '正在等待任务状态'
}

function friendlyNodeStatus(node) {
  const value = String(node || '').toLowerCase()
  if (value.includes('retrieve') || value.includes('knowledge') || value.includes('recall')) {
    return '正在匹配相关知识和指标口径'
  }
  if (value.includes('plan') || value.includes('graph') || value.includes('compile')) {
    return '正在规划查询路径'
  }
  if (value.includes('execute') || value.includes('query') || value.includes('doris') || value.includes('sql')) {
    return '正在查询经营数据'
  }
  if (value.includes('verify') || value.includes('evidence') || value.includes('critic')) {
    return '正在核对查询结果'
  }
  if (value.includes('answer')) {
    return '正在整理回答'
  }
  return '正在分析问题并读取经营数据'
}

function systemMessage(text) {
  return {
    id: `local_${Date.now()}`,
    answer: text,
    thinkingSteps: [],
    dorisTables: [],
    dataRows: [],
    dataSections: [],
    persisted: false
  }
}

function appendAssistant(response) {
  messages.value.push({
    localId: `a_${Date.now()}`,
    id: response.id || `local_${Date.now()}`,
    role: 'assistant',
    text: response.answer,
    steps: response.thinkingSteps || [],
    tables: (response.dorisTables || []).filter(table => table !== 'dim_merchant_df'),
    dataRows: response.dataRows || [],
    dataSections: (response.dataSections || []).map(section => ({
      ...section,
      dorisTables: (section.dorisTables || []).filter(table => table !== 'dim_merchant_df'),
      dataRows: section.dataRows || []
    })),
    feedbackStatus: {
      adopted: false,
      liked: false,
      disliked: false,
      persisted: Boolean(response.persisted)
    }
  })
  if (response.suggestions?.length) {
    suggestionPool.value = mergeSuggestionPool(response.suggestions)
    suggestionCursor.value = 0
    suggestions.value = suggestionPool.value.slice(0, 3)
  }
  if (response.context) {
    conversationContext.value = response.context
  }
  saveActiveSessionSnapshot()
}

async function handleFeedback(payload) {
  const message = messages.value.find(item => item.id === payload.id)
  if (message) {
    message.feedbackStatus = nextFeedbackStatus(message.feedbackStatus || {}, payload)
    saveActiveSessionSnapshot()
  }
  try {
    const result = await sendFeedback(payload.id, payload)
    if (message) {
      message.feedbackStatus.persisted = Boolean(result.persisted || message.feedbackStatus.persisted)
    }
  } catch {
    // 前端演示模式下忽略反馈写入失败。
  }
}

function nextFeedbackStatus(current, payload) {
  const next = { ...current }
  if (Object.prototype.hasOwnProperty.call(payload, 'adopted')) {
    next.adopted = Boolean(payload.adopted)
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'liked')) {
    next.liked = Boolean(payload.liked)
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'disliked')) {
    next.disliked = Boolean(payload.disliked)
  }
  if (next.liked) {
    next.disliked = false
  }
  if (next.disliked) {
    next.liked = false
  }
  return next
}

async function resetChat() {
  saveActiveSessionSnapshot()
  cancelActiveInteraction()
  input.value = ''
  newSessionFlash.value = true
  const session = createConversationSession('新会话', '您好，我是 Evan，新的经营分析会话已开启。')
  sessions.value = [session, ...sessions.value].slice(0, 6)
  loadSession(session.id)
  await nextTick()
  if (chatList.value) {
    chatList.value.scrollTo({ top: 0, behavior: 'smooth' })
  }
  inputRef.value?.focus()
  clearNewSessionTimer()
  newSessionTimer = window.setTimeout(() => {
    newSessionFlash.value = false
    newSessionTimer = null
  }, 1400)
}

async function switchSession(sessionId) {
  if (sessionId === activeSessionId.value) return
  saveActiveSessionSnapshot()
  cancelActiveInteraction()
  clearNewSessionTimer()
  newSessionFlash.value = false
  loadSession(sessionId)
  await nextTick()
  if (chatList.value) {
    chatList.value.scrollTo({ top: chatList.value.scrollHeight, behavior: 'smooth' })
  }
  inputRef.value?.focus()
}

function loadSession(sessionId) {
  const session = sessions.value.find(item => item.id === sessionId)
  if (!session) return
  activeSessionId.value = session.id
  messages.value = cloneValue(session.messages)
  conversationContext.value = cloneValue(session.conversationContext)
  suggestionPool.value = cloneValue(session.suggestionPool || defaultSuggestions)
  suggestionCursor.value = Number(session.suggestionCursor || 0)
  suggestions.value = cloneValue(session.suggestions || pickSuggestionPage(suggestionCursor.value))
  input.value = ''
  loading.value = false
  stopping.value = false
  runStatusText.value = '正在分析问题并读取经营数据'
}

function saveActiveSessionSnapshot() {
  const index = sessions.value.findIndex(item => item.id === activeSessionId.value)
  if (index < 0) return
  const existing = sessions.value[index]
  sessions.value[index] = {
    ...existing,
    title: sessionTitleFromMessages(messages.value, existing.title),
    messages: cloneValue(messages.value),
    conversationContext: cloneValue(conversationContext.value),
    suggestionPool: cloneValue(suggestionPool.value),
    suggestionCursor: suggestionCursor.value,
    suggestions: cloneValue(suggestions.value)
  }
}

function cancelActiveInteraction() {
  chatEpoch += 1
  clearRunPoll()
  if (submitController) {
    submitController.abort()
    clearSubmitController()
  }
  const running = activeRun.value
  activeRun.value = null
  if (running) {
    cancelRun(running.threadId, running.runId).catch(() => {})
  }
  loading.value = false
  stopping.value = false
}

function createConversationSession(title, welcomeText) {
  return {
    id: `session_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    title,
    messages: [createWelcomeMessage(welcomeText)],
    conversationContext: null,
    suggestionPool: defaultSuggestions.slice(),
    suggestionCursor: 0,
    suggestions: defaultSuggestions.slice(0, 3)
  }
}

function createWelcomeMessage(text) {
  return {
    localId: `welcome_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    role: 'assistant',
    text,
    steps: [],
    tables: [],
    dataRows: [],
    dataSections: [],
    feedbackStatus: {}
  }
}

function sessionTitleFromMessages(sessionMessages, fallback) {
  const lastUserMessage = [...sessionMessages].reverse().find(message => message.role === 'user')
  const text = String(lastUserMessage?.text || fallback || '新会话').trim()
  return text.length > 18 ? `${text.slice(0, 18)}...` : text
}

function cloneValue(value) {
  if (value === undefined || value === null) return value
  return JSON.parse(JSON.stringify(value))
}

function clearNewSessionTimer() {
  if (newSessionTimer) {
    window.clearTimeout(newSessionTimer)
    newSessionTimer = null
  }
}

function rotateSuggestions() {
  if (!suggestionPool.value.length) return
  if (suggestionPool.value.length <= suggestionPageSize) {
    suggestions.value = suggestionPool.value.slice()
    return
  }
  const current = new Set(suggestions.value)
  for (let attempts = 0; attempts < suggestionPool.value.length; attempts += 1) {
    const nextCursor = (suggestionCursor.value + suggestionPageSize + attempts) % suggestionPool.value.length
    const nextPage = pickSuggestionPage(nextCursor)
    if (nextPage.some(item => !current.has(item))) {
      suggestionCursor.value = nextCursor
      suggestions.value = nextPage
      return
    }
  }
  suggestionCursor.value = (suggestionCursor.value + 1) % suggestionPool.value.length
  suggestions.value = pickSuggestionPage(suggestionCursor.value)
}

function mergeSuggestionPool(serverSuggestions = []) {
  const merged = []
  for (const item of [...serverSuggestions, ...defaultSuggestions]) {
    const text = String(item || '').trim()
    if (!text || merged.includes(text)) continue
    merged.push(text)
  }
  return merged
}

function pickSuggestionPage(startIndex) {
  if (!suggestionPool.value.length) return []
  const page = []
  for (let offset = 0; offset < suggestionPageSize && offset < suggestionPool.value.length; offset += 1) {
    page.push(suggestionPool.value[(startIndex + offset) % suggestionPool.value.length])
  }
  return page
}

async function scrollBottom() {
  await nextTick()
  if (chatList.value) {
    chatList.value.scrollTo({ top: chatList.value.scrollHeight, behavior: 'smooth' })
    return
  }
  window.scrollTo({ top: document.documentElement.scrollHeight, behavior: 'smooth' })
}
</script>
