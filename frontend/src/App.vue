<template>
  <main class="app-shell">
    <header class="assistant-header">
      <img src="/yshopping-logo.svg" alt="yshopping" class="brand-logo" />
      <div class="brand-title">
        <h1>yshopping 商家 AI 助手</h1>
        <p>为你提供专业服务</p>
      </div>
      <button class="icon-button" type="button" title="新会话" @click="resetChat">
        <MessageCirclePlus :size="25" />
      </button>
    </header>

    <section class="hero-card">
      <p>我是 yshopping 商家助手，有问题我可以帮您解答，您是在日常经营中遇到问题了吗？</p>
    </section>

    <DailyReportCard v-if="dailyReport" :report="dailyReport" />

    <section class="chat-list" ref="chatList">
      <ChatMessage
        v-for="message in messages"
        :key="message.localId"
        :id="message.id"
        :role="message.role"
        :text="message.text"
        :steps="message.steps"
        :tables="message.tables"
        :data-rows="message.dataRows"
        :feedback-status="message.feedbackStatus"
        @feedback="handleFeedback"
      />
      <div v-if="loading" class="loading-card">
        <LoaderCircle :size="18" />
        <span>正在分析问题并读取经营数据</span>
      </div>
    </section>

    <SuggestionList
      :suggestions="suggestions"
      @select="sendSuggestion"
      @refresh="rotateSuggestions"
    />

    <form class="input-bar" @submit.prevent="submit">
      <input v-model.trim="input" type="text" placeholder="说出您的疑惑吧，yshopping 帮你解决" />
      <button type="submit" class="send-button" title="发送" :disabled="!input || loading">
        <Send :size="20" />
      </button>
    </form>
  </main>
</template>

<script setup>
import { nextTick, onMounted, ref } from 'vue'
import { LoaderCircle, MessageCirclePlus, Send } from 'lucide-vue-next'
import ChatMessage from './components/ChatMessage.vue'
import DailyReportCard from './components/DailyReportCard.vue'
import SuggestionList from './components/SuggestionList.vue'
import { getDailyReport, mockChat, mockDailyReport, sendFeedback, sendMessage } from './api/client'

const input = ref('')
const loading = ref(false)
const chatList = ref(null)
const dailyReport = ref(null)
const suggestions = ref(['我想查看保证金', '最近7天咨询工单量', '我要货品上架，具体规则有吗？'])
const suggestionPools = [
  ['我想查看保证金', '最近7天咨询工单量', '我要货品上架，具体规则有吗？'],
  ['昨天总 GMV 是多少？', '最近7天退货量趋势', '查看优惠券明细'],
  ['查看商品上架明细', '昨天退款金额是多少？', '我的商家手机号是多少？']
]
const messages = ref([
  {
    localId: 'welcome',
    role: 'assistant',
    text: '您好，我是 yshopping 商家 AI 助手，有经营问题欢迎随时问我。',
    steps: [],
    tables: [],
    dataRows: [],
    feedbackStatus: {}
  }
])

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
    const response = await sendMessage(message)
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
    feedbackStatus: {
      adopted: false,
      liked: false,
      disliked: false,
      persisted: Boolean(response.persisted)
    }
  })
  if (response.suggestions?.length) {
    suggestions.value = response.suggestions.slice(0, 3)
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
      feedbackStatus: {}
    }
  ]
}

function rotateSuggestions() {
  const current = suggestionPools.findIndex(pool => pool[0] === suggestions.value[0])
  suggestions.value = suggestionPools[(current + 1) % suggestionPools.length]
}

async function scrollBottom() {
  await nextTick()
  chatList.value?.scrollTo({ top: chatList.value.scrollHeight, behavior: 'smooth' })
}
</script>
