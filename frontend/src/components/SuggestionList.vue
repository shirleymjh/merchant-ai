<template>
  <section class="suggestion-card">
    <h2>猜你想问</h2>
    <button
      v-for="(item, index) in suggestions"
      :key="item"
      type="button"
      class="suggestion-row"
      @click="$emit('select', item)"
    >
      <span>{{ index + 1 }}</span>
      <b>{{ item }}</b>
      <ChevronRight :size="22" />
    </button>
    <button type="button" class="refresh-btn" @click="$emit('refresh')">
      <RefreshCw :size="18" />
      换一换
    </button>
  </section>
</template>

<script setup>
import { ChevronRight, RefreshCw } from 'lucide-vue-next'

defineProps({
  suggestions: {
    type: Array,
    default: () => []
  }
})

defineEmits(['select', 'refresh'])
</script>

