import {
  compactFixed,
  escapeHtml,
  humanizeIdentifier,
  isAsciiIdentifier,
  markdownLine,
  safeFileName,
  stripDatabaseQualifier,
  trimTrailingCharacters
} from '../utils/textParsing'

export function buildAnalysisReport(input = {}) {
  const experience = input.merchantExperience || {}
  const sections = normalizeSections(input.dataSections, input.dataRows, input.tables)
  const presentation = presentationMetadata(experience.metricDisclosures || [])
  const metricCards = buildMetricCards(sections, presentation)
  const trends = buildTrends(sections, presentation)
  const detailTables = buildDetailTables(sections, presentation)
  const traceability = experience.traceability || {}
  const sources = unique([
    ...(traceability.sourceTables || []),
    ...(input.tables || []),
    ...sections.flatMap(section => section.tables)
  ]).map(stripDatabaseQualifier).filter(Boolean)
  const definitions = (experience.metricDisclosures || []).map(item => ({
    name: item.displayName || item.metricKey || '指标口径',
    description: item.description || item.formula || '采用当前已发布的业务口径。'
  })).slice(0, 4)

  return {
    title: buildTitle(input.question, sections),
    question: String(input.question || '').trim(),
    generatedAt: formatDateTime(new Date()),
    timeRange: traceability.timeRange || '本次查询范围',
    dataUpdatedAt: traceability.dataUpdatedAt || '',
    evidenceStatus: traceability.evidenceStatus || '',
    summary: summarizeAnswer(input.answer),
    metricCards,
    trends,
    anomalies: (experience.anomalyAlerts || []).map(item => ({
      metric: item.metric || '经营指标',
      message: item.message || item.description || String(item)
    })).filter(item => item.message).slice(0, 4),
    actions: unique([
      ...(experience.businessAdvice || []),
      ...(experience.drillDownActions || []).map(item => item.label || item.question)
    ]).filter(Boolean).slice(0, 5),
    definitions,
    sources,
    detailTables
  }
}

export function hasAnalysisReportContent(report) {
  return Boolean(
    report?.metricCards?.length
    || report?.trends?.length
    || report?.anomalies?.length
    || report?.actions?.length
    || report?.detailTables?.length
  )
}

export function renderAnalysisReportHtml(report) {
  const metricCards = report.metricCards.map(card => `
    <article class="metric-card">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
      <small>${escapeHtml(card.context || report.timeRange)}</small>
    </article>`).join('')
  const trends = report.trends.map(trend => `
    <article class="panel trend-panel">
      <div class="panel-title"><span>趋势表现</span><h2>${escapeHtml(trend.title)}</h2></div>
      <div class="bars">${trend.points.map(point => `
        <div class="bar-item"><div class="bar-track"><i style="height:${point.height}%"></i></div><b>${escapeHtml(point.value)}</b><span>${escapeHtml(point.label)}</span></div>`).join('')}
      </div>
    </article>`).join('')
  const anomalyHtml = renderListPanel('异常与风险', report.anomalies.map(item => `<b>${escapeHtml(item.metric)}</b><span>${escapeHtml(item.message)}</span>`), 'risk')
  const actionHtml = renderListPanel('建议行动', report.actions.map((item, index) => `<b>${index + 1}</b><span>${escapeHtml(item)}</span>`), 'action')
  const definitionHtml = renderListPanel('指标口径', report.definitions.map(item => `<b>${escapeHtml(item.name)}</b><span>${escapeHtml(item.description)}</span>`), 'definition')
  const tableHtml = report.detailTables.map(table => `
    <article class="panel table-panel">
      <div class="panel-title"><span>数据明细</span><h2>${escapeHtml(table.title)}</h2></div>
      <div class="table-scroll"><table><thead><tr>${table.columns.map(column => `<th>${escapeHtml(column.label)}</th>`).join('')}</tr></thead>
      <tbody>${table.rows.map(row => `<tr>${table.columns.map(column => `<td>${escapeHtml(formatCell(row[column.key]))}</td>`).join('')}</tr>`).join('')}</tbody></table></div>
    </article>`).join('')

  return `<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>${escapeHtml(report.title)}</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#172033;font-family:Inter,"PingFang SC","Microsoft YaHei",sans-serif}.page{width:min(1120px,calc(100% - 32px));margin:32px auto 64px}.hero{position:relative;overflow:hidden;padding:48px;border-radius:28px;color:#fff;background:linear-gradient(135deg,#15243e 0%,#203e67 65%,#27618a 100%);box-shadow:0 24px 64px rgba(22,43,72,.2)}.hero:after{content:"";position:absolute;width:280px;height:280px;border-radius:50%;right:-80px;top:-120px;background:rgba(74,213,181,.16)}.brand{font-size:13px;letter-spacing:.12em;color:#74ddc0}.hero h1{max-width:760px;margin:20px 0 12px;font-size:38px;line-height:1.18}.hero p{max-width:800px;margin:0;color:#d9e5f3;line-height:1.7}.meta{display:flex;gap:18px;flex-wrap:wrap;margin-top:28px;color:#b8c9dc;font-size:13px}.metric-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin:18px 0}.metric-card,.panel{background:#fff;border:1px solid #e3e8ef;border-radius:20px;box-shadow:0 10px 35px rgba(38,55,77,.06)}.metric-card{padding:22px}.metric-card span,.metric-card small{display:block;color:#708095;font-size:13px}.metric-card strong{display:block;margin:11px 0 8px;font-size:27px;color:#15243e}.panel{padding:26px;margin-top:16px}.panel-title span{font-size:12px;letter-spacing:.12em;color:#268b78}.panel-title h2{margin:6px 0 20px;font-size:21px}.summary{font-size:16px;line-height:1.85;color:#3f4d60}.two-column{display:grid;grid-template-columns:1fr 1fr;gap:16px}.item-list{display:grid;gap:10px}.list-item{display:grid;grid-template-columns:auto 1fr;gap:12px;align-items:start;padding:14px;border-radius:14px;background:#f7f9fb}.list-item b{color:#1c806f}.risk .list-item{background:#fff7f1}.risk .list-item b{color:#c05c2f}.list-item span{line-height:1.55;color:#445269}.bars{height:220px;display:flex;align-items:flex-end;gap:10px;padding-top:20px}.bar-item{flex:1;min-width:40px;text-align:center}.bar-track{height:145px;display:flex;align-items:flex-end;border-radius:8px;background:#edf3f7;overflow:hidden}.bar-track i{display:block;width:100%;min-height:4px;background:linear-gradient(180deg,#48c9aa,#2384a7);border-radius:8px 8px 0 0}.bar-item b,.bar-item span{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.bar-item b{margin-top:7px;font-size:12px}.bar-item span{margin-top:4px;color:#7e8b9c;font-size:11px}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:12px 14px;text-align:left;border-bottom:1px solid #e9edf2;white-space:nowrap}th{color:#647388;background:#f7f9fb}.foot{margin-top:18px;padding:20px;color:#6e7c8f;font-size:12px;line-height:1.7}.foot b{color:#344257}@media(max-width:760px){.page{width:min(100% - 20px,1120px);margin-top:10px}.hero{padding:30px 24px}.hero h1{font-size:29px}.metric-grid{grid-template-columns:repeat(2,1fr)}.two-column{grid-template-columns:1fr}.panel{padding:20px}}@media print{body{background:#fff}.page{width:100%;margin:0}.hero,.metric-card,.panel{box-shadow:none;break-inside:avoid}}
</style></head><body><main class="page">
  <header class="hero"><div class="brand">YSHOPPING · MERCHANT INTELLIGENCE</div><h1>${escapeHtml(report.title)}</h1><p>${escapeHtml(report.summary || '基于本轮经营数据生成的结构化分析报告。')}</p><div class="meta"><span>分析周期：${escapeHtml(report.timeRange)}</span><span>生成时间：${escapeHtml(report.generatedAt)}</span>${report.dataUpdatedAt ? `<span>数据更新：${escapeHtml(report.dataUpdatedAt)}</span>` : ''}</div></header>
  ${metricCards ? `<section class="metric-grid">${metricCards}</section>` : ''}
  ${trends}
  <section class="two-column">${anomalyHtml}${actionHtml}</section>
  ${tableHtml}
  ${definitionHtml}
  <footer class="foot"><b>数据说明</b><br>数据来源：${escapeHtml(report.sources.join('、') || '本轮已查询的经营数据')}。本报告由当前会话已有结果生成，未为展示重复查询数据。内容为 AI 生成，仅供经营决策参考。</footer>
</main></body></html>`
}

export function downloadAnalysisReport(report) {
  const blob = new Blob([renderAnalysisReportHtml(report)], { type: 'text/html;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = `${safeFileName(report.title)}.html`
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function normalizeSections(dataSections = [], dataRows = [], tables = []) {
  const sections = (dataSections || []).map(section => ({
    title: section.title || section.resultSummary || '经营数据',
    rows: (section.dataRows || []).filter(Boolean),
    tables: section.dorisTables || [],
    columnLabels: section.columnLabels || {},
    columnFormats: section.columnFormats || {}
  })).filter(section => section.rows.length)
  if (sections.length) return sections
  return (dataRows || []).length ? [{ title: '经营数据', rows: dataRows, tables: tables || [] }] : []
}

function buildMetricCards(sections, presentation) {
  const cards = []
  for (const section of sections) {
    const rows = section.rows
    if (rows.every(isTrendRow)) {
      const groups = groupBy(rows, row => row.metric_name || section.title)
      for (const [metric, points] of Object.entries(groups)) {
        const latest = points[points.length - 1]
        cards.push({ label: metricLabel(metric, presentation, section), value: formatMetric(latest.value, metric, presentation), context: `最新值 · ${latest.pt || ''}` })
      }
      continue
    }
    if (rows.length === 1) {
      for (const [key, value] of Object.entries(rows[0])) {
        if (key.startsWith('__') || isIdentifier(key) || !isNumeric(value)) continue
        cards.push({ label: metricLabel(key, presentation, section), value: formatMetric(value, key, presentation), context: section.title })
      }
    }
  }
  return dedupeBy(cards, card => card.label).slice(0, 4)
}

function buildTrends(sections, presentation) {
  return sections.filter(section => section.rows.every(isTrendRow)).flatMap(section => {
    const groups = groupBy(section.rows, row => row.metric_name || section.title)
    return Object.entries(groups).map(([metric, rows]) => {
      const latest = rows.slice(-8)
      const values = latest.map(row => Number(row.value) || 0)
      const max = Math.max(...values.map(Math.abs), 1)
      return {
        title: metricLabel(metric, presentation, section),
        points: latest.map((row, index) => ({
          label: shortDate(row.pt),
          value: formatMetric(values[index], metric, presentation),
          height: Math.max(4, Math.round(Math.abs(values[index]) / max * 100))
        }))
      }
    })
  }).slice(0, 2)
}

function buildDetailTables(sections, presentation) {
  return sections.filter(section => !section.rows.every(isTrendRow) && section.rows.length).map(section => {
    const keys = unique(section.rows.flatMap(row => Object.keys(row || {}))).filter(key => !key.startsWith('__')).slice(0, 8)
    return {
      title: section.title || '经营明细',
      columns: keys.map(key => ({ key, label: metricLabel(key, presentation, section) })),
      rows: section.rows.slice(0, 10)
    }
  }).slice(0, 2)
}

function renderListPanel(title, items, kind) {
  if (!items.length) return ''
  return `<article class="panel ${kind}"><div class="panel-title"><span>经营解读</span><h2>${title}</h2></div><div class="item-list">${items.map(item => `<div class="list-item">${item}</div>`).join('')}</div></article>`
}

function summarizeAnswer(answer) {
  const paragraphs = String(answer || '')
    .split('\n')
    .map(markdownLine)
    .filter(line => line.kind === 'text' || line.kind === 'bullet')
    .map(line => line.text)
    .filter(Boolean)
  return paragraphs.slice(0, 3).join(' ').slice(0, 420)
}

function buildTitle(question, sections) {
  const text = trimTrailingCharacters(String(question || '').trim(), '？?。.')
  if (text) return text.length > 34 ? `${text.slice(0, 34)}…` : `${text}｜经营分析报告`
  return `${sections[0]?.title || '商家经营'}｜经营分析报告`
}

function metricLabel(value, presentation, section = {}) {
  const raw = String(value || '').trim()
  if (!raw) return '经营指标'
  return section.columnLabels?.[raw]
    || presentation[raw.toLowerCase()]?.displayName
    || (isAsciiIdentifier(raw) ? humanizeIdentifier(raw) : raw)
}

function formatMetric(value, key, presentation) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return formatCell(value)
  const metadata = presentation[String(key || '').toLowerCase()] || {}
  const scale = Number(metadata.scale ?? 1)
  const displayed = numeric * (Number.isFinite(scale) ? scale : 1)
  const compact = Math.abs(displayed) >= 10000
    ? `${compactFixed(displayed / 10000)}万`
    : displayed.toLocaleString('zh-CN', { maximumFractionDigits: 2 })
  return `${metadata.prefix || ''}${compact}${metadata.unit || metadata.suffix || ''}`
}

function formatCell(value) {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

function isTrendRow(row) {
  return row && Object.prototype.hasOwnProperty.call(row, 'pt') && Object.prototype.hasOwnProperty.call(row, 'value')
}

function isIdentifier(key) {
  const value = String(key).toLowerCase()
  return value === 'pt' || value.endsWith('_id') || value.endsWith('_no')
}

function isNumeric(value) {
  return value !== '' && value !== null && value !== undefined && Number.isFinite(Number(value))
}

function groupBy(items, selector) {
  return items.reduce((result, item) => {
    const key = selector(item)
    result[key] = result[key] || []
    result[key].push(item)
    return result
  }, {})
}

function unique(items) {
  return [...new Set(items.filter(Boolean))]
}

function dedupeBy(items, selector) {
  const seen = new Set()
  return items.filter(item => {
    const key = selector(item)
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
}

function shortDate(value) {
  const text = String(value || '')
  return text.length >= 10 ? text.slice(5, 10) : text
}

function presentationMetadata(disclosures) {
  const result = {}
  for (const item of disclosures || []) {
    const key = String(item?.metricKey || '').trim().toLowerCase()
    if (!key) continue
    result[key] = {
      displayName: item.displayName || item.metricKey,
      unit: item.unit || '',
      prefix: item.prefix || '',
      suffix: item.suffix || '',
      scale: item.scale ?? 1
    }
  }
  return result
}

function formatDateTime(date) {
  return new Intl.DateTimeFormat('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false }).format(date)
}
