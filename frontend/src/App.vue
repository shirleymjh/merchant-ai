<template>
  <main class="diana-app">
    <aside class="workspace-sidebar">
      <div class="sidebar-brand">
        <img src="/yshopping-logo.svg" alt="yshopping" />
        <div>
          <span>当前经营主题</span>
          <strong>商家经营分析</strong>
        </div>
      </div>
      <div class="sidebar-search">
        <Search :size="15" />
        <input v-model.trim="sessionFilter" type="search" placeholder="搜索常用问题" />
      </div>
      <nav class="session-list">
        <button class="session-item active" type="button">
          <span class="session-dot"></span>
          <span>当前会话</span>
          <small>当前</small>
        </button>
        <button
          v-for="item in filteredSessionExamples"
          :key="item"
          class="session-item"
          type="button"
          @click="sendSuggestion(item)"
        >
          <span class="session-dot muted-dot"></span>
          <span>{{ item }}</span>
          <small>常用</small>
        </button>
        <p v-if="!filteredSessionExamples.length" class="session-empty">没有匹配的问题</p>
      </nav>
    </aside>

    <section class="workspace-main">
      <header class="topic-header">
        <div class="topic-tabs">
          <button class="topic-tab active" type="button">
            <span>{{ currentTopicTitle }}</span>
          </button>
          <button class="topic-tab new-session" type="button" @click="resetChat">
            <Plus :size="15" />
            <span>新会话</span>
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
          <span>正在分析问题并读取经营数据</span>
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
          <input v-model.trim="input" type="text" placeholder="说出您的疑惑吧，yshopping 帮你解决" />
          <button type="submit" class="send-button" title="发送" :disabled="!input || loading">
            <Send :size="20" />
          </button>
        </form>
      </div>
    </section>

    <aside class="insight-rail">
      <DailyReportCard v-if="dailyReport" :report="dailyReport" :compact="true" />
      <section class="rail-card">
        <p class="rail-kicker">分析能力</p>
        <h3>经营数据分析</h3>
        <p>可查经营指标、明细数据、规则说明，并基于结果做异常判断。</p>
      </section>
    </aside>
  </main>
</template>

<script setup>
import { computed, nextTick, onMounted, ref } from 'vue'
import { LoaderCircle, Plus, Search, Send, Zap } from 'lucide-vue-next'
import ChatMessage from './components/ChatMessage.vue'
import DailyReportCard from './components/DailyReportCard.vue'
import SuggestionList from './components/SuggestionList.vue'
import { getDailyReport, mockChat, mockDailyReport, sendFeedback, sendMessage } from './api/client'

const input = ref('')
const loading = ref(false)
const chatList = ref(null)
const dailyReport = ref(null)
const sessionFilter = ref('')
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
const sessionExamples = [
  '最近7天咨询工单量',
  '最近30天 GMV 下降原因',
  '最近10天退款明细',
  '商品审核拒绝原因',
  '保证金充值流水'
]
const suggestions = ref(defaultSuggestions.slice(0, 3))
const suggestionPool = ref(defaultSuggestions.slice())
const suggestionCursor = ref(0)
const suggestionPageSize = 3
const conversationContext = ref(null)
const messages = ref([
  {
    localId: 'welcome',
    role: 'assistant',
    text: '您好，我是 yshopping 商家 AI 助手，有经营问题欢迎随时问我。',
    steps: [],
    tables: [],
    dataRows: [],
    dataSections: [],
    feedbackStatus: {}
  }
])

const hasConversation = computed(() => messages.value.some(message => message.role === 'user'))
const currentTopicTitle = computed(() => {
  const lastUserMessage = [...messages.value].reverse().find(message => message.role === 'user')
  return lastUserMessage?.text || '经营分析工作台'
})
const filteredSessionExamples = computed(() => {
  const keyword = sessionFilter.value.trim()
  if (!keyword) return sessionExamples
  return sessionExamples.filter(item => item.includes(keyword))
})

onMounted(async () => {
  try {
    dailyReport.value = await getDailyReport()
  } catch {
    dailyReport.value = mockDailyReport()
  }
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
  messages.value.push({
    localId: `u_${Date.now()}`,
    role: 'user',
    text: message
  })
  loading.value = true
  await scrollBottom()
  try {
    const response = await sendMessage(message, conversationContext.value)
    appendAssistant(response)
  } catch {
    appendAssistant(mockChat(message))
  } finally {
    loading.value = false
    await scrollBottom()
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
}

async function handleFeedback(payload) {
  const message = messages.value.find(item => item.id === payload.id)
  if (message) {
    message.feedbackStatus = nextFeedbackStatus(message.feedbackStatus || {}, payload)
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
  if (payload.adopted) next.adopted = true
  if (payload.liked) {
    next.liked = true
    next.disliked = false
  }
  if (payload.disliked) {
    next.disliked = true
    next.liked = false
  }
  return next
}

function resetChat() {
  messages.value = [
    {
      localId: 'welcome',
      role: 'assistant',
      text: '您好，我是 yshopping 商家 AI 助手，有经营问题欢迎随时问我。',
      steps: [],
      tables: [],
      dataRows: [],
      dataSections: [],
      feedbackStatus: {}
    }
  ]
  conversationContext.value = null
  suggestionPool.value = defaultSuggestions.slice()
  suggestionCursor.value = 0
  suggestions.value = pickSuggestionPage(0)
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
