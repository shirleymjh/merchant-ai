import { createApp } from 'vue'
import App from './App.vue'
import './assets/styles.css'

if (
  import.meta.env.DEV
  && new URLSearchParams(globalThis.location.search).get('internalPreview') === '1'
  && !globalThis.__MERCHANT_AI_RUNTIME__
) {
  globalThis.__MERCHANT_AI_RUNTIME__ = {
    internalMode: true,
    opsActor: 'local-preview',
    identity: { userId: 'local-preview', displayName: '本地预览' }
  }
}

createApp(App).mount('#app')
