import { readdirSync, readFileSync, statSync } from 'node:fs'
import { dirname, extname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { parse } from '@babel/parser'

const frontendRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const repositoryRoot = dirname(frontendRoot)
const sourceRoot = join(frontendRoot, 'src')
const topicRoot = join(repositoryRoot, 'python_backend', 'resources', 'runtime', 'topics')
const supportedExtensions = new Set(['.js', '.mjs', '.vue'])
const semanticIdentifierKeys = new Set(['metricKey', 'tableName', 'topic'])
const violations = []

function sourceFiles(directory, extensions = supportedExtensions) {
  const files = []
  for (const entry of readdirSync(directory).sort()) {
    const path = join(directory, entry)
    if (statSync(path).isDirectory()) {
      files.push(...sourceFiles(path, extensions))
    } else if (extensions.has(extname(path))) {
      files.push(path)
    }
  }
  return files
}

function collectSemanticIdentifiers(value, output) {
  if (Array.isArray(value)) {
    for (const item of value) collectSemanticIdentifiers(item, output)
    return
  }
  if (!value || typeof value !== 'object') return
  for (const [key, item] of Object.entries(value)) {
    if (semanticIdentifierKeys.has(key) && typeof item === 'string' && item) output.add(item)
    collectSemanticIdentifiers(item, output)
  }
}

function publishedSemanticIdentifiers() {
  const identifiers = new Set()
  for (const entry of readdirSync(topicRoot).sort()) {
    const path = join(topicRoot, entry)
    if (statSync(path).isDirectory()) identifiers.add(entry)
  }
  for (const file of sourceFiles(topicRoot, new Set(['.json']))) {
    collectSemanticIdentifiers(JSON.parse(readFileSync(file, 'utf8')), identifiers)
  }
  return identifiers
}

function vueScriptBlocks(source) {
  const blocks = []
  let cursor = 0
  while (cursor < source.length) {
    const opening = source.indexOf('<script', cursor)
    if (opening < 0) break
    const contentStart = source.indexOf('>', opening)
    if (contentStart < 0) break
    const closing = source.indexOf('</script>', contentStart + 1)
    if (closing < 0) break
    blocks.push({ source: source.slice(contentStart + 1, closing), offset: contentStart + 1 })
    cursor = closing + '</script>'.length
  }
  return blocks
}

function isRegExpConstructor(node) {
  if (!node) return false
  if (node.type === 'Identifier') return node.name === 'RegExp'
  return Boolean(
    node.type === 'MemberExpression' &&
      !node.computed &&
      node.property?.type === 'Identifier' &&
      node.property.name === 'RegExp'
  )
}

function inspectNode(node, file, governedIdentifiers) {
  if (!node || typeof node !== 'object') return
  if (node.type === 'RegExpLiteral' || (node.type === 'Literal' && node.regex)) {
    violations.push(`${file}:${node.loc?.start?.line || 1}: regular-expression literal`)
  }
  if (
    (node.type === 'CallExpression' || node.type === 'NewExpression') &&
    isRegExpConstructor(node.callee)
  ) {
    violations.push(`${file}:${node.loc?.start?.line || 1}: RegExp constructor`)
  }
  if (node.type === 'StringLiteral' && governedIdentifiers.has(node.value)) {
    violations.push(
      `${file}:${node.loc?.start?.line || 1}: published business identifier ${node.value}`
    )
  }
  for (const [key, value] of Object.entries(node)) {
    if (key === 'loc' || key === 'start' || key === 'end') continue
    if (Array.isArray(value)) {
      for (const item of value) inspectNode(item, file, governedIdentifiers)
    } else if (value && typeof value === 'object') {
      inspectNode(value, file, governedIdentifiers)
    }
  }
}

const governedIdentifiers = publishedSemanticIdentifiers()
for (const file of sourceFiles(sourceRoot)) {
  const source = readFileSync(file, 'utf8')
  const blocks = extname(file) === '.vue' ? vueScriptBlocks(source) : [{ source, offset: 0 }]
  for (const block of blocks) {
    const ast = parse(block.source, {
      sourceType: 'module',
      plugins: ['importAttributes'],
    })
    inspectNode(ast, file, governedIdentifiers)
  }
}

if (violations.length) {
  process.stderr.write(`Frontend source governance violations:\n${violations.join('\n')}\n`)
  process.exit(1)
}

process.stdout.write('Frontend no-regex and no-business-identifier contract passed.\n')
