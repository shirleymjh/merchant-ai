export function isWhitespace(character) {
  return Boolean(character) && character.trim() === ''
}

export function collapseWhitespace(value, replacement = ' ') {
  let result = ''
  let pending = false
  for (const character of String(value ?? '')) {
    if (isWhitespace(character)) {
      pending = result.length > 0
      continue
    }
    if (pending) result += replacement
    result += character
    pending = false
  }
  return result
}

export function replaceAllLiteral(value, target, replacement = '') {
  const source = String(value ?? '')
  const literal = String(target ?? '')
  return literal ? source.split(literal).join(replacement) : source
}

export function replaceCharacters(value, characters, replacement = '') {
  const blocked = new Set(Array.from(String(characters ?? '')))
  let result = ''
  for (const character of String(value ?? '')) {
    result += blocked.has(character) ? replacement : character
  }
  return result
}

export function trimTrailingCharacters(value, characters) {
  const source = String(value ?? '')
  const removable = new Set(Array.from(String(characters ?? '')))
  let end = source.length
  while (end > 0 && removable.has(source[end - 1])) end -= 1
  return source.slice(0, end)
}

export function isAsciiIdentifier(value) {
  const source = String(value ?? '')
  if (!source) return false
  const first = source.charCodeAt(0)
  if (!isAsciiLetter(first) && source[0] !== '_') return false
  for (let index = 1; index < source.length; index += 1) {
    const code = source.charCodeAt(index)
    if (!isAsciiLetter(code) && !isAsciiDigit(code) && source[index] !== '_') return false
  }
  return true
}

export function humanizeIdentifier(value) {
  const source = String(value ?? '').trim()
  if (!source) return ''
  let result = ''
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index]
    const code = source.charCodeAt(index)
    if (character === '_' || isWhitespace(character)) {
      if (result && !result.endsWith(' ')) result += ' '
      continue
    }
    const previousCode = index > 0 ? source.charCodeAt(index - 1) : 0
    if (isAsciiUpper(code) && (isAsciiLower(previousCode) || isAsciiDigit(previousCode)) && !result.endsWith(' ')) {
      result += ' '
    }
    result += character
  }
  return collapseWhitespace(result).trim()
}

export function stripDatabaseQualifier(value) {
  const parts = String(value ?? '').trim().split('.').filter(Boolean)
  return parts.length ? parts[parts.length - 1] : ''
}

export function compactFixed(value, digits = 2) {
  let text = Number(value).toFixed(digits)
  while (text.endsWith('0')) text = text.slice(0, -1)
  if (text.endsWith('.')) text = text.slice(0, -1)
  return text
}

export function safeFileName(value, fallback = 'result', limit = 70) {
  const invalid = new Set(Array.from('\\/:*?"<>|'))
  let result = ''
  let pendingSeparator = false
  for (const character of String(value || fallback).trim()) {
    if (invalid.has(character) || isWhitespace(character)) {
      pendingSeparator = result.length > 0
      continue
    }
    if (pendingSeparator && !result.endsWith('_')) result += '_'
    result += character
    pendingSeparator = false
    if (result.length >= limit) break
  }
  return result || fallback
}

export function escapeHtml(value) {
  const entities = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  }
  let result = ''
  for (const character of String(value ?? '')) result += entities[character] || character
  return result
}

export function markdownLine(value) {
  let source = String(value ?? '').trim()
  if (!source) return { kind: 'empty', text: '' }
  let hashes = 0
  while (hashes < source.length && source[hashes] === '#') hashes += 1
  if (hashes > 0 && hashes <= 6 && isWhitespace(source[hashes])) {
    return { kind: 'heading', text: cleanInlineMarkdown(source.slice(hashes).trim()) }
  }
  if (source[0] === '>' && isWhitespace(source[1])) source = source.slice(1).trim()
  if (new Set(['-', '*', '•']).has(source[0]) && isWhitespace(source[1])) {
    return { kind: 'bullet', text: cleanInlineMarkdown(source.slice(1).trim()) }
  }
  return { kind: 'text', text: cleanInlineMarkdown(source) }
}

export function cleanInlineMarkdown(value) {
  return replaceAllLiteral(replaceAllLiteral(String(value ?? ''), '**'), '`').trim()
}

export function delimitedContents(value, opening, closing) {
  const source = String(value ?? '')
  const results = []
  let cursor = 0
  while (cursor < source.length) {
    const start = source.indexOf(opening, cursor)
    if (start < 0) break
    const contentStart = start + opening.length
    const end = source.indexOf(closing, contentStart)
    if (end < 0) break
    const content = source.slice(contentStart, end).trim()
    if (content) results.push(content)
    cursor = end + closing.length
  }
  return results
}

export function isoDateParts(value) {
  const source = String(value ?? '')
  for (let index = 0; index + 10 <= source.length; index += 1) {
    const candidate = source.slice(index, index + 10)
    if (candidate[4] !== '-' || candidate[7] !== '-') continue
    const digits = candidate.slice(0, 4) + candidate.slice(5, 7) + candidate.slice(8, 10)
    if (Array.from(digits).every(character => isAsciiDigit(character.charCodeAt(0)))) {
      return [candidate.slice(0, 4), candidate.slice(5, 7), candidate.slice(8, 10)]
    }
  }
  return []
}

export function pathSegment(value, prefix, index) {
  const source = String(value ?? '')
  if (!source.startsWith(prefix)) return ''
  return source.slice(prefix.length).split('/').filter(Boolean)[index] || ''
}

function isAsciiDigit(code) {
  return code >= 48 && code <= 57
}

function isAsciiUpper(code) {
  return code >= 65 && code <= 90
}

function isAsciiLower(code) {
  return code >= 97 && code <= 122
}

function isAsciiLetter(code) {
  return isAsciiUpper(code) || isAsciiLower(code)
}
