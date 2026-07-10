<template>
  <main class="evan-app">
    <section class="workspace-main">
      <header class="topic-header">
        <div class="topic-tabs">
          <button
            v-for="session in sessions"
            :key="session.id"
            :class="['topic-tab', { active: session.id === activeSessionId, running: isSessionRunning(session) }]"
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
          :merchant-experience="message.merchantExperience"
          :feedback-status="message.feedbackStatus"
          @feedback="handleFeedback"
          @ask="sendSuggestion"
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
      <DailyReportCard v-if="dailyReport" :report="dailyReport" :compact="true" @ask="sendSuggestion" />
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
const defaultRunStatusText = '正在分析问题并读取经营数据'
const sessionStorageKey = 'evan_merchant_ai_sessions_v1'
const runPollTimers = new Map()
let newSessionTimer = null
const chatList = ref(null)
const inputRef = ref(null)
const dailyReport = ref(null)
const newSessionFlash = ref(false)
const defaultSuggestions = [
  '最近7天店铺整体经营情况怎么样？',
  '最近7天订单量和退款金额有什么变化？',
  '最近30天 GMV 为什么下降？',
  '退款金额最高的前5个商品有哪些？',
  '退款率高的商品是否也带来较多工单？',
  '工单最多的问题类型有哪些？',
  '最近7天客服工单量按天趋势如何？',
  '最近10天商品审核拒绝明细',
  '最近7天履约量和发货超时订单量',
  '最近30天赔付金额最高的订单有哪些？',
  '最近7天优惠金额和 GMV 表现如何？',
  '保证金余额和冻结金额是否异常？',
  '最近30天申诉次数和处罚次数',
  '商品审核被拒后优先排查什么？',
  '催单工单最近是否升高？'
]
const suggestionPageSize = 3
const initialSession = createConversationSession('经营分析工作台', '您好，我是 yshopping 商家经营助手。可以帮您查订单、退款售后、客服工单、商品审核、履约和经营趋势。')
const restoredConversation = restorePersistedSessions(initialSession)
const sessions = ref(restoredConversation.sessions)
const activeSessionId = ref(restoredConversation.activeSessionId)
const initialVisibleSession = findSessionSnapshot(sessions.value, activeSessionId.value) || sessions.value[0]
const messages = ref(cloneValue(initialVisibleSession.messages))
const suggestions = ref(cloneValue(initialVisibleSession.suggestions || defaultSuggestions.slice(0, 3)))
const suggestionPool = ref(cloneValue(initialVisibleSession.suggestionPool || defaultSuggestions.slice()))
const suggestionCursor = ref(Number(initialVisibleSession.suggestionCursor || 0))
const conversationContext = ref(cloneValue(initialVisibleSession.conversationContext))

const activeSession = computed(() => sessions.value.find(session => session.id === activeSessionId.value) || sessions.value[0])
const hasConversation = computed(() => messages.value.some(message => message.role === 'user'))
const loading = computed(() => isSessionRunning(activeSession.value))
const stopping = computed(() => Boolean(activeSession.value?.stopping))
const runStatusText = computed(() => activeSession.value?.runStatusText || defaultRunStatusText)
onMounted(async () => {
  resumeSessionRuns()
  try {
    dailyReport.value = await getDailyReport()
  } catch {
    dailyReport.value = mockDailyReport()
  }
})

onBeforeUnmount(() => {
  clearAllRunPolls()
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
  const session = activeSession.value
  if (!session || isSessionRunning(session)) return
  const sessionId = session.id
  const controller = new AbortController()
  clearNewSessionTimer()
  newSessionFlash.value = false
  messages.value.push({
    localId: `u_${Date.now()}`,
    role: 'user',
    text: message
  })
  saveActiveSessionSnapshot()
  updateSessionRuntime(sessionId, {
    submitting: true,
    submitController: controller,
    stopping: false,
    stopRequested: false,
    runStatusText: '正在提交任务'
  })
  await scrollBottom()
  try {
    const requestContext = cloneValue(session.conversationContext)
    const requestHistory = buildMessageHistory(messages.value)
    const created = await startAsyncRun(message, requestContext, { signal: controller.signal, messageHistory: requestHistory })
    const runId = created.runId
    const threadId = created.threadId
    if (!runId || !threadId) {
      throw new Error('RUN_CREATE_FAILED')
    }
    const run = {
      runId,
      threadId,
      token: `run_${Date.now()}_${Math.random().toString(16).slice(2)}`
    }
    updateSessionRuntime(sessionId, {
      activeRun: run,
      submitting: false,
      submitController: null,
      runStatusText: '任务已提交，正在排队'
    })
    scheduleRunPoll(sessionId, run.token, 300)
  } catch (error) {
    const aborted = error?.name === 'AbortError'
    const currentSession = findSession(sessionId)
    updateSessionRuntime(sessionId, {
      activeRun: null,
      submitting: false,
      submitController: null,
      stopping: false,
      runStatusText: defaultRunStatusText
    })
    if (aborted) {
      if (currentSession?.stopRequested) {
        appendAssistantToSession(sessionId, systemMessage('已停止本次回答。您可以修改问题后重新提问。'))
      }
    } else {
      appendAssistantToSession(sessionId, mockChat(message))
    }
  }
}

function scheduleRunPoll(sessionId, token, delay = 900) {
  clearRunPoll(sessionId)
  const timer = window.setTimeout(() => {
    pollSessionRun(sessionId, token)
  }, delay)
  runPollTimers.set(sessionId, timer)
}

async function pollSessionRun(sessionId, token) {
  const session = findSession(sessionId)
  const current = session?.activeRun
  if (!current || current.token !== token) return
  try {
    const [runPayload, eventsPayload] = await Promise.allSettled([
      getRun(current.threadId, current.runId),
      getRunEvents(current.threadId, current.runId)
    ])
    const latestSession = findSession(sessionId)
    if (!latestSession?.activeRun || latestSession.activeRun.token !== token) return
    const run = runPayload.status === 'fulfilled' ? runPayload.value?.run : null
    let nextStatusText = defaultRunStatusText
    if (eventsPayload.status === 'fulfilled') {
      nextStatusText = latestRunStatusText(run, eventsPayload.value?.events || [])
    } else {
      nextStatusText = latestRunStatusText(run, [])
    }
    updateSessionRuntime(sessionId, { runStatusText: nextStatusText })
    const status = String(run?.status || '').toUpperCase()
    if (status === 'COMPLETED') {
      clearRunPoll(sessionId)
      updateSessionRuntime(sessionId, {
        activeRun: null,
        submitting: false,
        stopping: false,
        runStatusText: defaultRunStatusText
      })
      if (run.answer) {
        appendAssistantToSession(sessionId, run.answer)
      } else {
        appendAssistantToSession(sessionId, systemMessage('任务已完成，但没有返回可展示的答案。'))
      }
      return
    }
    if (status === 'FAILED') {
      clearRunPoll(sessionId)
      updateSessionRuntime(sessionId, {
        activeRun: null,
        submitting: false,
        stopping: false,
        runStatusText: defaultRunStatusText
      })
      appendAssistantToSession(sessionId, systemMessage(`本次回答失败：${run?.error || '后端执行异常'}`))
      return
    }
    if (status === 'CANCELED') {
      clearRunPoll(sessionId)
      updateSessionRuntime(sessionId, {
        activeRun: null,
        submitting: false,
        stopping: false,
        runStatusText: defaultRunStatusText
      })
      return
    }
    scheduleRunPoll(sessionId, token)
  } catch {
    const latestSession = findSession(sessionId)
    if (!latestSession?.activeRun || latestSession.activeRun.token !== token) return
    updateSessionRuntime(sessionId, { runStatusText: '正在等待后端返回状态' })
    scheduleRunPoll(sessionId, token, 1200)
  }
}

async function stopCurrentRun() {
  if (stopping.value) return
  const session = activeSession.value
  if (!session) return
  const current = session.activeRun
  updateSessionRuntime(session.id, {
    stopping: true,
    stopRequested: true,
    runStatusText: '正在停止本次回答'
  })
  clearRunPoll(session.id)
  if (!current && session.submitController) {
    session.submitController.abort()
    return
  }
  if (!current) {
    updateSessionRuntime(session.id, {
      submitting: false,
      stopping: false,
      submitController: null,
      runStatusText: defaultRunStatusText
    })
    return
  }
  const stoppedRun = current
  updateSessionRuntime(session.id, { activeRun: null })
  try {
    await cancelRun(stoppedRun.threadId, stoppedRun.runId)
  } catch {
    // 取消请求失败也要允许用户继续提问；后端旧结果不会再被当前 token 接收。
  } finally {
    updateSessionRuntime(session.id, {
      submitting: false,
      stopping: false,
      runStatusText: defaultRunStatusText
    })
    appendAssistantToSession(session.id, systemMessage('已停止本次回答。您可以修改问题后重新提问。'))
  }
}

function clearRunPoll(sessionId) {
  const timer = runPollTimers.get(sessionId)
  if (timer) {
    window.clearTimeout(timer)
    runPollTimers.delete(sessionId)
  }
}

function clearAllRunPolls() {
  for (const timer of runPollTimers.values()) {
    window.clearTimeout(timer)
  }
  runPollTimers.clear()
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
  appendAssistantToSession(activeSessionId.value, response)
}

function appendAssistantToSession(sessionId, response) {
  const targetMessages = sessionId === activeSessionId.value ? messages.value : cloneValue(findSession(sessionId)?.messages || [])
  targetMessages.push({
    localId: `a_${Date.now()}_${Math.random().toString(16).slice(2)}`,
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
    merchantExperience: response.merchantExperience || response.merchant_experience || {},
    feedbackStatus: {
      adopted: false,
      liked: false,
      disliked: false,
      persisted: Boolean(response.persisted)
    }
  })
  const sessionUpdates = {
    messages: cloneValue(targetMessages)
  }
  if (response.suggestions?.length) {
    const nextPool = mergeSuggestionPoolFor(sessionId, response.suggestions)
    sessionUpdates.suggestionPool = nextPool
    sessionUpdates.suggestionCursor = 0
    sessionUpdates.suggestions = nextPool.slice(0, 3)
  }
  if (response.context) {
    sessionUpdates.conversationContext = cloneValue(response.context)
  }
  updateSession(sessionId, sessionUpdates)
  if (sessionId === activeSessionId.value) {
    messages.value = cloneValue(sessionUpdates.messages)
    if (sessionUpdates.suggestionPool) {
      suggestionPool.value = cloneValue(sessionUpdates.suggestionPool)
      suggestionCursor.value = 0
      suggestions.value = cloneValue(sessionUpdates.suggestions)
    }
    if (response.context) {
      conversationContext.value = cloneValue(response.context)
    }
    scrollBottom()
  }
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
  input.value = ''
  newSessionFlash.value = true
  const session = createConversationSession('新会话', '您好，我是 yshopping 商家经营助手。新的经营分析会话已开启。')
  sessions.value = [session, ...sessions.value].slice(0, 6)
  loadSession(session.id)
  persistSessions()
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
  persistSessions()
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
  persistSessions()
}

function createConversationSession(title, welcomeText) {
  return {
    id: `session_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    title,
    messages: [createWelcomeMessage(welcomeText)],
    conversationContext: null,
    suggestionPool: defaultSuggestions.slice(),
    suggestionCursor: 0,
    suggestions: defaultSuggestions.slice(0, 3),
    activeRun: null,
    submitting: false,
    submitController: null,
    stopping: false,
    stopRequested: false,
    runStatusText: defaultRunStatusText
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
    merchantExperience: {},
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

function buildMessageHistory(sessionMessages) {
  return (sessionMessages || [])
    .filter(message => ['user', 'assistant'].includes(message.role) && String(message.text || '').trim())
    .slice(-16)
    .map(message => ({
      id: message.id || '',
      localId: message.localId || '',
      role: message.role,
      text: String(message.text || '').slice(0, 1200)
    }))
}

function restorePersistedSessions(fallbackSession) {
  if (typeof window === 'undefined' || !window.localStorage) {
    return { sessions: [fallbackSession], activeSessionId: fallbackSession.id }
  }
  try {
    const raw = window.localStorage.getItem(sessionStorageKey)
    if (!raw) return { sessions: [fallbackSession], activeSessionId: fallbackSession.id }
    const parsed = JSON.parse(raw)
    const restoredSessions = Array.isArray(parsed?.sessions)
      ? parsed.sessions.map(normalizeRestoredSession).filter(Boolean).slice(0, 6)
      : []
    if (!restoredSessions.length) {
      return { sessions: [fallbackSession], activeSessionId: fallbackSession.id }
    }
    const restoredActiveId = restoredSessions.some(session => session.id === parsed.activeSessionId)
      ? parsed.activeSessionId
      : restoredSessions[0].id
    return { sessions: restoredSessions, activeSessionId: restoredActiveId }
  } catch {
    return { sessions: [fallbackSession], activeSessionId: fallbackSession.id }
  }
}

function normalizeRestoredSession(session) {
  if (!session || typeof session !== 'object') return null
  const activeRun = session.activeRun?.threadId && session.activeRun?.runId
    ? {
        threadId: session.activeRun.threadId,
        runId: session.activeRun.runId,
        token: session.activeRun.token || `run_restore_${Date.now()}_${Math.random().toString(16).slice(2)}`
      }
    : null
  return {
    id: session.id || `session_restore_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    title: session.title || '经营分析工作台',
    messages: Array.isArray(session.messages) && session.messages.length
      ? cloneValue(session.messages)
      : [createWelcomeMessage('您好，我是 yshopping 商家经营助手。新的经营分析会话已开启。')],
    conversationContext: cloneValue(session.conversationContext),
    suggestionPool: Array.isArray(session.suggestionPool) && session.suggestionPool.length
      ? cloneValue(session.suggestionPool)
      : defaultSuggestions.slice(),
    suggestionCursor: Number(session.suggestionCursor || 0),
    suggestions: Array.isArray(session.suggestions) && session.suggestions.length
      ? cloneValue(session.suggestions)
      : defaultSuggestions.slice(0, 3),
    activeRun,
    submitting: false,
    submitController: null,
    stopping: false,
    stopRequested: false,
    runStatusText: activeRun ? (session.runStatusText || '正在等待任务状态') : defaultRunStatusText
  }
}

function findSessionSnapshot(sessionList, sessionId) {
  return sessionList.find(session => session.id === sessionId)
}

function persistSessions() {
  if (typeof window === 'undefined' || !window.localStorage) return
  const payload = {
    activeSessionId: activeSessionId.value,
    sessions: sessions.value.map(toPersistedSession)
  }
  window.localStorage.setItem(sessionStorageKey, JSON.stringify(payload))
}

function toPersistedSession(session) {
  return {
    ...session,
    messages: cloneValue(session.messages),
    conversationContext: cloneValue(session.conversationContext),
    suggestionPool: cloneValue(session.suggestionPool),
    suggestions: cloneValue(session.suggestions),
    submitController: null,
    submitting: false,
    stopping: false,
    stopRequested: false
  }
}

function resumeSessionRuns() {
  for (const session of sessions.value) {
    if (session.activeRun?.token) {
      scheduleRunPoll(session.id, session.activeRun.token, 250)
    }
  }
}

function clearNewSessionTimer() {
  if (newSessionTimer) {
    window.clearTimeout(newSessionTimer)
    newSessionTimer = null
  }
}

function findSession(sessionId) {
  return sessions.value.find(session => session.id === sessionId)
}

function isSessionRunning(session) {
  return Boolean(session?.submitting || session?.activeRun)
}

function updateSession(sessionId, updates) {
  const index = sessions.value.findIndex(session => session.id === sessionId)
  if (index < 0) return
  sessions.value[index] = {
    ...sessions.value[index],
    ...updates
  }
  persistSessions()
}

function updateSessionRuntime(sessionId, updates) {
  updateSession(sessionId, updates)
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
      saveActiveSessionSnapshot()
      return
    }
  }
  suggestionCursor.value = (suggestionCursor.value + 1) % suggestionPool.value.length
  suggestions.value = pickSuggestionPage(suggestionCursor.value)
  saveActiveSessionSnapshot()
}

function mergeSuggestionPool(serverSuggestions = []) {
  return mergeSuggestionPoolFrom(defaultSuggestions, serverSuggestions)
}

function mergeSuggestionPoolFor(sessionId, serverSuggestions = []) {
  const session = findSession(sessionId)
  return mergeSuggestionPoolFrom(session?.suggestionPool || defaultSuggestions, serverSuggestions)
}

function mergeSuggestionPoolFrom(basePool, serverSuggestions = []) {
  const merged = []
  const defaultSuggestionSet = new Set(defaultSuggestions.map(item => String(item || '').trim()).filter(Boolean))
  const dynamicSuggestions = serverSuggestions
    .map(item => String(item || '').trim())
    .filter(Boolean)
  const cleanedBasePool = (basePool || [])
    .map(item => String(item || '').trim())
    .filter(Boolean)
  const source = dynamicSuggestions.length
    ? [...dynamicSuggestions, ...cleanedBasePool.filter(item => !defaultSuggestionSet.has(item))]
    : [...cleanedBasePool, ...defaultSuggestions]
  for (const item of source) {
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
