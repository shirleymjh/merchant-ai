<template>
  <section v-if="suggestions.length" :class="['suggestion-card', { compact }]">
    <div class="suggestion-head">
      <h2>猜你想问</h2>
      <button type="button" class="refresh-btn" @click="$emit('refresh')">
        <RefreshCw :size="16" />
        换一批
      </button>
    </div>
    <div class="suggestion-chip-list">
      <button
        v-for="(item, index) in suggestions"
        :key="item"
        type="button"
        class="suggestion-chip"
        @click="$emit('select', item)"
      >
        <span class="suggestion-index">{{ index + 1 }}</span>
        <b>{{ item }}</b>
      </button>
    </div>
  </section>
</template>

<script setup>
import { RefreshCw } from 'lucide-vue-next'

defineProps({
  suggestions: {
    type: Array,
    default: () => []
  },
  compact: {
    type: Boolean,
    default: false
  }
})

defineEmits(['select', 'refresh'])
</script>
