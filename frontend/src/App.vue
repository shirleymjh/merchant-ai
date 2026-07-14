<template>
  <main :class="['evan-app', { 'compact-workspace': !showAnalysisRail }]">
    <header class="app-brand-header">
      <button type="button" class="brand-menu" title="打开对话目录" @click="outlineOpen = true"><PanelLeftOpen :size="20" /></button>
      <div class="brand-mark"><ShoppingBag :size="24" /></div>
      <div class="brand-copy"><h1>yshopping 商家 AI 助手</h1><p>经营数据、分析与行动建议</p></div>
      <label class="identity-selector" title="选择当前使用角色">
        <span>当前身份</span>
        <select v-model="userIdentity.role">
          <option v-if="internalMode" value="platform_operator">平台运营管理员</option>
          <option value="merchant_owner">店铺负责人</option>
          <option value="merchant_operator">经营运营</option>
          <option value="merchant_finance">财务</option>
          <option value="merchant_customer_service">客服</option>
          <option value="merchant_goods">商品运营</option>
          <option value="merchant_fulfillment">履约运营</option>
        </select>
      </label>
      <button v-if="internalMode" type="button" class="brand-admin" title="打开内部经营配置" @click="governanceOpen = true"><Settings2 :size="17" />经营配置</button>
      <button type="button" class="brand-new-chat" @click="resetChat"><MessageCirclePlus :size="17" />新会话</button>
    </header>
    <div v-if="outlineOpen" class="outline-backdrop" @click="outlineOpen = false" />
    <aside :class="['outline-drawer', { open: outlineOpen }]" aria-label="对话目录">
      <div class="outline-drawer-head">
        <div><span>当前会话</span><h2>对话目录</h2></div>
        <button type="button" title="关闭目录" @click="outlineOpen = false"><PanelLeftClose :size="20" /></button>
      </div>
      <div class="outline-drawer-list">
        <button type="button" :class="{ active: !conversationOutline.length }" @click="scrollConversationTop">
          <b>1</b><span>开始对话</span><ChevronRight :size="17" />
        </button>
        <button v-for="(item, index) in conversationOutline" :key="item.id" type="button" :class="{ active: index === conversationOutline.length - 1 }" @click="openOutlineItem(item.id)">
          <b>{{ index + 2 }}</b><span>{{ item.label }}</span><ChevronRight :size="17" />
        </button>
      </div>
      <p class="outline-drawer-tip"><MessageSquareText :size="15" />选择一轮对话，相关图表与经营建议会同步更新。</p>
    </aside>
    <aside v-if="showAnalysisRail" class="analysis-rail">
      <div class="rail-title-row">
        <div class="rail-icon blue"><ChartNoAxesCombined :size="17" /></div>
        <div><span>数据洞察</span><strong>指标图表</strong></div>
      </div>
      <MetricInsightPanel :message="latestAssistantMessage" />
    </aside>
    <section class="workspace-main">
      <section class="chat-canvas" ref="chatList">
        <section v-if="!hasConversation" class="welcome-banner">
          <div class="welcome-icon"><Sparkles :size="22" /></div>
          <div><h2>您好，我是您的经营助手</h2><p>可以查询经营指标、查看业务明细，也可以结合数据给出分析与建议。</p></div>
        </section>
        <DailyReportCard v-if="dailyReport && !hasConversation" :report="dailyReport" @ask="sendSuggestion" />
        <section v-if="!hasConversation" class="upload-guide-card">
          <strong>您可以直接提问，也可以上传经营报表、截图或 PDF，我会结合内容进行分析。</strong>
          <span>内容为 AI 生成，仅供参考</span>
        </section>
        <nav v-if="conversationOutline.length" class="conversation-outline" aria-label="当前对话目录">
          <span><ListTree :size="14" /> 本轮目录</span>
          <button v-for="item in conversationOutline" :key="item.id" type="button" @click="scrollToMessage(item.id)">
            {{ item.label }}
          </button>
        </nav>
        <ChatMessage
          v-for="message in visibleMessages"
          :key="message.localId"
          :id="message.id"
          :anchor-id="message.localId"
          :role="message.role"
          :text="message.text"
          :question="message.question"
          :steps="message.steps"
          :tables="message.tables"
          :data-rows="message.dataRows"
          :data-sections="message.dataSections"
          :merchant-experience="message.merchantExperience"
          :clarification="message.clarification"
          :feedback-status="message.feedbackStatus"
          :workspace-mode="true"
          @feedback="handleFeedback"
          @ask="sendSuggestion"
          @confirm-clarification="handleClarificationConfirm"
          @metric-definition-action="handleMetricDefinitionAction"
          @knowledge-proposal-action="handleKnowledgeProposalAction"
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
        <div v-if="attachments.length" class="attachment-strip">
          <span v-for="(file, index) in attachments" :key="`${file.name}-${index}`" class="attachment-chip">
            <FileText :size="14" /><b>{{ file.name }}</b><button type="button" title="移除附件" @click="removeAttachment(index)"><X :size="12" /></button>
          </span>
        </div>
        <form class="input-bar" @submit.prevent="submit">
          <input ref="fileInput" class="file-input" type="file" multiple accept="image/*,.pdf,.xls,.xlsx,.csv" @change="handleFiles" />
          <button type="button" class="attach-button" title="上传图片、Excel、PDF 或 CSV" @click="fileInput?.click()"><Paperclip :size="19" /></button>
          <textarea ref="inputRef" v-model="input" rows="1" placeholder="输入经营问题，或上传报表、截图让我分析" @input="resizeComposer" @keydown.enter.exact.prevent="submit" />
          <button v-if="!loading" type="submit" class="send-button" title="发送" :disabled="!input">
            <Send :size="20" />
          </button>
          <button v-else type="button" class="send-button stop-button" title="停止回答" :disabled="stopping" @click="stopCurrentRun">
            <CircleStop :size="20" />
          </button>
        </form>
        <div class="composer-hint"><span>Enter 发送 · Shift + Enter 换行</span><span>支持图片、PDF、Excel、CSV</span></div>
      </div>
    </section>

    <aside class="insight-rail">
      <div class="rail-title-row">
        <div class="rail-icon amber"><Lightbulb :size="17" /></div>
        <div><span>结合本轮数据</span><strong>经营行动</strong></div>
      </div>
      <section v-if="currentAdvice.length" class="action-list">
        <article v-for="(advice, index) in currentAdvice" :key="advice">
          <span>{{ index + 1 }}</span><div><b>{{ adviceTitle(advice) }}</b><p>{{ advice }}</p><button type="button" :class="{ done: executedAdvice.has(index) }" @click="toggleAdvice(index)">{{ executedAdvice.has(index) ? '已标记执行' : '标记执行' }}</button></div>
        </article>
      </section>
      <section v-else class="rail-empty-state"><Lightbulb :size="25" /><b>等待分析结果</b><p>提问指标后，这里会给出至少两条可执行建议。</p></section>
      <div v-if="dataFreshness" class="data-freshness"><DatabaseZap :size="14" /><span>数据更新：{{ dataFreshness }}</span><b>已校验</b></div>
    </aside>
    <GovernanceConsole v-if="governanceOpen" @close="governanceOpen = false" />
  </main>
</template>

<script setup>
import { computed, nextTick, onBeforeUnmount, onMounted, ref } from 'vue'
import { ChartNoAxesCombined, ChevronRight, CircleStop, DatabaseZap, FileText, Lightbulb, ListTree, LoaderCircle, MessageCirclePlus, MessageSquareText, PanelLeftClose, PanelLeftOpen, Paperclip, Send, Settings2, ShoppingBag, Sparkles, X } from 'lucide-vue-next'
import ChatMessage from './components/ChatMessage.vue'
import DailyReportCard from './components/DailyReportCard.vue'
import MetricInsightPanel from './components/MetricInsightPanel.vue'
import GovernanceConsole from './components/GovernanceConsole.vue'
import SuggestionList from './components/SuggestionList.vue'
import { actOnKnowledgeSuggestion, cancelRun, getDailyReport, getMerchantProfile, getRun, getRunEvents, recordMetricDefinitionPreference, resumeChatRun, sendFeedback, streamChatRun, uploadAttachment } from './api/client'

const input = ref('')
const defaultRunStatusText = '正在分析问题并读取经营数据'
const sessionStorageKey = 'evan_merchant_ai_sessions_v1'
const runPollTimers = new Map()
let newSessionTimer = null
const chatList = ref(null)
const inputRef = ref(null)
const fileInput = ref(null)
const attachments = ref([])
const outlineOpen = ref(false)
const governanceOpen = ref(false)
const internalMode = new URLSearchParams(window.location.search).get('ops') === '1'
const userIdentity = ref({
  userId: internalMode ? 'platform_ops' : 'merchant_user_100',
  displayName: internalMode ? '平台运营管理员' : '当前商家用户',
  role: internalMode ? 'platform_operator' : 'merchant_operator',
  region: 'CN', language: 'zh-CN', storeIds: [], permissions: internalMode ? ['merchant.read', 'governance.write'] : ['merchant.read']
})
const executedAdvice = ref(new Set())
const dailyReport = ref(null)
const merchantProfile = ref(null)
const newSessionFlash = ref(false)
const defaultSuggestions = [
  '最近7天店铺整体经营情况怎么样？',
  '最近7天订单量和退款金额有什么变化？',
  '最近30天 GMV 为什么下降？',
  '最近30天退款金额最高的前5个商品有哪些？',
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
const visibleMessages = computed(() => hasConversation.value ? messages.value : [])
const latestAssistantMessage = computed(() => [...messages.value].reverse().find(message => message.role === 'assistant') || null)
const showAnalysisRail = computed(() => hasConversation.value && hasMetricInsightChart(latestAssistantMessage.value))
const currentAdvice = computed(() => {
  const advice = latestAssistantMessage.value?.merchantExperience?.businessAdvice || []
  const fallback = latestAssistantMessage.value?.merchantExperience?.drillDownActions?.map(item => item.label || item.question) || []
  const answerAdvice = extractAdviceFromAnswer(latestAssistantMessage.value?.text || '')
  const suggested = latestAssistantMessage.value?.suggestions || []
  const candidates = advice.length >= 2 ? advice : [...advice, ...answerAdvice, ...fallback, ...suggested]
  return candidates
    .filter(Boolean)
    .filter((item, index, all) => all.indexOf(item) === index)
    .slice(0, 2)
})
const dataFreshness = computed(() => latestAssistantMessage.value?.merchantExperience?.traceability?.dataUpdatedAt || '')
const conversationOutline = computed(() => messages.value.filter(message => message.role === 'user').slice(-5).map((message, index) => ({
  id: message.localId,
  label: String(message.text || `问题 ${index + 1}`).slice(0, 18)
})))
const loading = computed(() => isSessionRunning(activeSession.value))
const stopping = computed(() => Boolean(activeSession.value?.stopping))
const runStatusText = computed(() => activeSession.value?.runStatusText || defaultRunStatusText)
onMounted(async () => {
  resumeSessionRuns()
  const [reportResult, profileResult] = await Promise.allSettled([getDailyReport(), getMerchantProfile()])
  dailyReport.value = reportResult.status === 'fulfilled' ? reportResult.value : null
  merchantProfile.value = profileResult.status === 'fulfilled' ? profileResult.value.profile || null : null
})

onBeforeUnmount(() => {
  clearAllRunPolls()
  clearNewSessionTimer()
})

async function submit() {
  if (!input.value || loading.value) return
  const message = input.value
  input.value = ''
  const files = attachments.value.slice()
  attachments.value = []
  await nextTick()
  resizeComposer()
  await askWithOptions(message, { attachments: files })
}

async function sendSuggestion(question) {
  if (loading.value) return
  await ask(question)
}

async function ask(message) {
  return askWithOptions(message, {})
}

async function askWithOptions(message, options = {}) {
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
    const requestContext = cloneValue(options.context || session.conversationContext)
    const requestHistory = buildMessageHistory(messages.value)
    const requestThreadId = options.resumeThreadId || session.serverThreadId || ''
    const requestOptions = {
      signal: controller.signal,
      messageHistory: requestHistory,
      threadId: requestThreadId,
      attachments: options.attachments || [],
      userIdentity: cloneValue(userIdentity.value)
    }
    if (!options.resumeThreadId) {
      const completed = await streamChatRun(message, requestContext, requestOptions, event => handleStreamEvent(sessionId, event))
      if (!completed?.response) throw new Error('STREAM_COMPLETED_WITHOUT_RESPONSE')
      const completedThreadId = completed.threadId || requestThreadId || findSession(sessionId)?.serverThreadId || ''
      updateSessionRuntime(sessionId, {
        activeRun: null,
        ...(completedThreadId ? { serverThreadId: completedThreadId } : {}),
        submitting: false,
        submitController: null,
        stopping: false,
        runStatusText: defaultRunStatusText
      })
      appendAssistantToSession(sessionId, completed.response, completed.runId)
      return
    }
    const created = await resumeChatRun(message, requestContext, requestOptions)
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
      serverThreadId: threadId,
      submitting: false,
      submitController: null,
      runStatusText: '任务已提交，正在排队'
    })
    scheduleRunPoll(sessionId, run.token, 300)
  } catch (error) {
    const aborted = error?.name === 'AbortError'
    const currentSession = findSession(sessionId)
    const streamingRunId = currentSession?.activeRun?.runId || ''
    updateSessionRuntime(sessionId, {
      activeRun: null,
      submitting: false,
      submitController: null,
      stopping: false,
      runStatusText: defaultRunStatusText
    })
    if (aborted) {
      return
    } else {
      const detail = String(error?.message || '后端执行异常').slice(0, 160)
      appendAssistantToSession(sessionId, systemMessage(`本次回答失败：${detail}。请稍后重试。`), streamingRunId)
    }
  }
}

function handleStreamEvent(sessionId, event) {
  if (event.event === 'answer.delta') {
    appendStreamingDelta(sessionId, event.payload?.runId || event.runId || '', event.payload?.delta || '')
    return
  }
  if (event.event === 'run.started') {
    const threadId = event.payload?.threadId || event.threadId || ''
    updateSessionRuntime(sessionId, {
      activeRun: {
        runId: event.payload?.runId || event.runId,
        threadId,
        token: `stream_${Date.now()}`
      },
      ...(threadId ? { serverThreadId: threadId } : {}),
      submitting: false,
      runStatusText: '正在分析问题'
    })
    return
  }
  if (event.node) updateSessionRuntime(sessionId, { runStatusText: friendlyNodeStatus(event.node) })
}

function resizeComposer() {
  const element = inputRef.value
  if (!element) return
  element.style.height = 'auto'
  element.style.height = `${Math.min(Math.max(element.scrollHeight, 42), 180)}px`
}

async function handleFiles(event) {
  const files = Array.from(event.target.files || []).slice(0, 6)
  const prepared = await Promise.all(files.map(async file => {
    try {
      const uploaded = await uploadAttachment(file)
      return { id: uploaded.attachmentId, name: file.name, type: file.type || 'application/octet-stream', size: file.size }
    } catch {
      return { name: file.name, type: file.type || 'application/octet-stream', size: file.size, uploadFailed: true }
    }
  }))
  attachments.value = [...attachments.value, ...prepared].slice(0, 6)
  event.target.value = ''
}

function removeAttachment(index) { attachments.value.splice(index, 1) }
function adviceTitle(text) {
  if (/退款|售后/.test(text)) return '控制退款风险'
  if (/订单|转化|GMV|成交/.test(text)) return '提升交易表现'
  if (/客服|工单|响应/.test(text)) return '优化服务体验'
  return '建议立即关注'
}
function hasMetricInsightChart(message) {
  if (!message) return false
  const sections = message.dataSections?.length ? message.dataSections : [{ dataRows: message.dataRows || [] }]
  return sections.some(section => {
    const rows = section.dataRows || []
    const timeRows = rows
      .map(row => ({ pt: row.pt || row.date || row.dt, value: Number(row.value ?? row.metric_value ?? row.cnt) }))
      .filter(row => row.pt && Number.isFinite(row.value))
    if (timeRows.length > 1) return true
    const dimensionRows = rows
      .map(row => ({ label: String(row.group_value ?? row.name ?? row.category ?? ''), value: Number(row.metric_value ?? row.value ?? row.cnt) }))
      .filter(row => row.label && Number.isFinite(row.value))
    return dimensionRows.length > 1
  })
}
function extractAdviceFromAnswer(text) {
  const lines = String(text || '').split('\n').map(line => line.trim())
  const start = lines.findIndex(line => /^建议[:：]?$/.test(line))
  if (start < 0) return []
  return lines.slice(start + 1).filter(line => /^[-•]/.test(line)).map(line => line.replace(/^[-•]\s*/, '')).slice(0, 3)
}
function scrollToMessage(id) {
  document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}
function openOutlineItem(id) { outlineOpen.value = false; nextTick(() => scrollToMessage(id)) }
function scrollConversationTop() { outlineOpen.value = false; nextTick(() => chatList.value?.scrollTo({ top: 0, behavior: 'smooth' })) }
function toggleAdvice(index) {
  const next = new Set(executedAdvice.value)
  next.has(index) ? next.delete(index) : next.add(index)
  executedAdvice.value = next
}

async function handleClarificationConfirm(payload) {
  const answer = String(payload?.value || '').trim()
  if (!answer || loading.value) return
  const session = activeSession.value
  const threadId = payload?.threadId || payload?.checkpoint?.threadId || payload?.checkpoint?.checkpointThreadId || session?.activeRun?.threadId || session?.serverThreadId || ''
  await askWithOptions(answer, {
    resumeThreadId: threadId,
    context: cloneValue(session?.conversationContext)
  })
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
        serverThreadId: current.threadId,
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
        serverThreadId: current.threadId,
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
        serverThreadId: current.threadId,
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
    updateSessionRuntime(session.id, {
      submitting: false,
      stopping: false,
      submitController: null,
      runStatusText: defaultRunStatusText
    })
    appendAssistantToSession(session.id, systemMessage('已停止本次回答。您可以修改问题后重新提问。'))
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
    session.submitController?.abort()
    await cancelRun(stoppedRun.threadId, stoppedRun.runId)
  } catch {
    // 取消请求失败也要允许用户继续提问；后端旧结果不会再被当前 token 接收。
  } finally {
    updateSessionRuntime(session.id, {
      submitting: false,
      stopping: false,
      runStatusText: defaultRunStatusText
    })
    appendAssistantToSession(
      session.id,
      systemMessage('已停止本次回答。您可以修改问题后重新提问。'),
      stoppedRun.runId
    )
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

function appendAssistantToSession(sessionId, response, streamingRunId = '') {
  const targetMessages = sessionId === activeSessionId.value ? messages.value : cloneValue(findSession(sessionId)?.messages || [])
  const sourceQuestion = [...targetMessages].reverse().find(message => message.role === 'user')?.text || ''
  const nextMessage = {
    localId: `a_${Date.now()}_${Math.random().toString(16).slice(2)}`,
    id: response.id || `local_${Date.now()}`,
    role: 'assistant',
    question: sourceQuestion,
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
    suggestions: response.suggestions || [],
    clarification: response.clarification || null,
    feedbackStatus: {
      adopted: false,
      liked: false,
      disliked: false,
      persisted: Boolean(response.persisted)
    }
  }
  const streamingIndex = streamingRunId
    ? targetMessages.findIndex(message => message.streamingRunId === streamingRunId)
    : -1
  if (streamingIndex >= 0) {
    nextMessage.localId = targetMessages[streamingIndex].localId
    targetMessages.splice(streamingIndex, 1, nextMessage)
  } else {
    targetMessages.push(nextMessage)
  }
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

function appendStreamingDelta(sessionId, runId, delta) {
  const text = String(delta || '')
  if (!runId || !text) return
  const targetMessages = sessionId === activeSessionId.value ? messages.value : cloneValue(findSession(sessionId)?.messages || [])
  let message = targetMessages.find(item => item.streamingRunId === runId)
  if (!message) {
    const sourceQuestion = [...targetMessages].reverse().find(item => item.role === 'user')?.text || ''
    message = {
      localId: `stream_${runId}`,
      role: 'assistant',
      question: sourceQuestion,
      text: '',
      steps: [],
      tables: [],
      dataRows: [],
      dataSections: [],
      merchantExperience: {},
      suggestions: [],
      clarification: null,
      feedbackStatus: {},
      streamingRunId: runId
    }
    targetMessages.push(message)
  }
  message.text += text
  const nextMessages = cloneValue(targetMessages)
  updateSession(sessionId, { messages: nextMessages })
  if (sessionId === activeSessionId.value) {
    messages.value = nextMessages
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

async function handleMetricDefinitionAction(payload) {
  const message = messages.value.find(item => item.id === payload.answerId)
  if (message) {
    const experience = { ...(message.merchantExperience || {}) }
    experience.metricDefinitionPreference = {
      status: payload.action === 'question' ? 'question_recorded' : 'preference_recorded',
      metricKey: payload.metricKey,
      updatedAt: new Date().toISOString()
    }
    message.merchantExperience = experience
    saveActiveSessionSnapshot()
  }
  try {
    await recordMetricDefinitionPreference({
      ...payload,
      reviewer: userIdentity.value.displayName || userIdentity.value.userId || 'merchant_user'
    })
  } catch {
    // 前端演示模式下忽略偏好写入失败。
  }
}

async function handleKnowledgeProposalAction(payload) {
  const message = messages.value.find(item => item.id === payload.answerId)
  if (!message || !payload.suggestionId) return
  const updateProposal = (patch) => {
    const experience = { ...(message.merchantExperience || {}) }
    experience.knowledgeSuggestions = (experience.knowledgeSuggestions || []).map(item =>
      item.suggestionId === payload.suggestionId ? { ...item, ...patch } : item
    )
    message.merchantExperience = experience
    saveActiveSessionSnapshot()
  }
  updateProposal({ actionPending: true, actionError: '' })
  try {
    const result = await actOnKnowledgeSuggestion(payload.suggestionId, payload.action, {
      actor: userIdentity.value.displayName || userIdentity.value.userId || 'merchant_user',
      conflictResolution: payload.conflictResolution || ''
    })
    const suggestion = result.suggestion || {}
    if (String(result.status || '').toUpperCase() === 'CONFLICT_CONFIRMATION_REQUIRED') {
      updateProposal({
        conflictCheck: result.conflictCheck || suggestion.payload?.conflictCheck || {},
        allowedActions: result.allowedActions || undefined,
        actionPending: false,
        actionError: ''
      })
      return
    }
    updateProposal({
      status: suggestion.status || String(result.status || '').toLowerCase(),
      scopeType: suggestion.scopeType || result.scopeType || payload.scopeType,
      merchantAction: suggestion.merchantAction || payload.action,
      conflictCheck: suggestion.payload?.conflictCheck || null,
      actionPending: false,
      actionError: ''
    })
  } catch (error) {
    updateProposal({ actionPending: false, actionError: `操作失败：${error?.message || error}` })
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
    serverThreadId: '',
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
    serverThreadId: session.serverThreadId || activeRun?.threadId || '',
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
    messages: compactPersistedMessages(session.messages),
    conversationContext: cloneValue(session.conversationContext),
    suggestionPool: cloneValue(session.suggestionPool),
    suggestions: cloneValue(session.suggestions),
    submitController: null,
    submitting: false,
    stopping: false,
    stopRequested: false
  }
}

function compactPersistedMessages(sessionMessages) {
  return (sessionMessages || []).slice(-20).map(message => ({
    ...message,
    text: String(message.text || '').slice(0, 6000),
    steps: (message.steps || []).slice(-12),
    dataRows: (message.dataRows || []).slice(0, 60),
    dataSections: (message.dataSections || []).slice(0, 4).map(section => ({
      ...section,
      dataRows: (section.dataRows || []).slice(0, 60)
    })),
    merchantExperience: compactMerchantExperience(message.merchantExperience || {})
  }))
}

function compactMerchantExperience(experience) {
  return {
    businessAdvice: (experience.businessAdvice || []).slice(0, 4),
    anomalyAlerts: (experience.anomalyAlerts || []).slice(0, 4),
    drillDownActions: (experience.drillDownActions || []).slice(0, 4),
    metricDisclosures: (experience.metricDisclosures || []).slice(0, 4),
    metricDefinitionPreference: experience.metricDefinitionPreference || {},
    traceability: experience.traceability || {}
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
