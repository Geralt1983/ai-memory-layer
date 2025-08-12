'use client'
import { useEffect, useMemo } from 'react'
import { marked } from 'marked'
import DOMPurify from 'dompurify'
import hljs from 'highlight.js'
import 'highlight.js/styles/github-dark.css'

const PRETTY_LANG: Record<string, string> = {
  tsx: 'TSX', ts: 'TypeScript', js: 'JavaScript', jsx: 'JSX',
  json: 'JSON', py: 'Python', rb: 'Ruby', rs: 'Rust',
  go: 'Go', java: 'Java', c: 'C', cpp: 'C++', cs: 'C#',
  sh: 'Shell', bash: 'Bash', zsh: 'Zsh', yaml: 'YAML', yml: 'YAML',
  md: 'Markdown', html: 'HTML', css: 'CSS', sql: 'SQL', php: 'PHP',
}

function prettyLang(lang?: string) {
  if (!lang) return ''
  const key = lang.toLowerCase()
  return PRETTY_LANG[key] || lang
}

// Parse info string: "ts {lines start=5}"
function parseInfo(info?: string) {
  const raw = (info || '').trim()
  if (!raw) return { lang: '', lines: false, start: 1 }
  const m = raw.match(/^([^\s{]+)?\s*(\{.*\})?$/)
  const lang = (m?.[1] || '').trim()
  let lines = false
  let start = 1
  if (m?.[2]) {
    const meta = m[2]
    if (/\blines\b/.test(meta)) lines = true
    const sm = meta.match(/start\s*=\s*(\d+)/)
    if (sm) start = Math.max(1, parseInt(sm[1], 10) || 1)
  }
  return { lang, lines, start }
}

// Custom renderer with ChatGPT style code blocks, optional line numbers
const renderer = new marked.Renderer()
renderer.code = ({ text, lang }: { text: string; lang?: string }) => {
  const { lang: parsedLang, lines, start } = parseInfo(lang)
  const label = prettyLang(parsedLang)
  let highlighted = text
  try {
    if (parsedLang && hljs.getLanguage(parsedLang)) {
      highlighted = hljs.highlight(text, { language: parsedLang }).value
    } else {
      highlighted = hljs.highlightAuto(text).value
    }
  } catch { /* noop */ }

  // If line numbers requested, wrap each line with <span>..</span>
  let bodyHtml = highlighted
  let lineStyle = ''
  let blockClass = 'codeblock'
  if (lines) {
    const parts = highlighted.split('\n')
    // Preserve trailing newline behavior
    const lastEmpty = parts.length && parts[parts.length - 1] === ''
    const wrapped = parts.map(l => `<span>${l}</span>`).join('\n')
    bodyHtml = lastEmpty ? `${wrapped}\n` : wrapped
    blockClass += ' with-lines'
    // counter starts at start-1 because ::before increments first
    lineStyle = `style="counter-reset: line ${start - 1}"`
  }

  const langAttr = parsedLang ? `language-${parsedLang}` : ''
  return `
<div class="${blockClass}" data-lang="${label}">
  <div class="codeblock-header">
    <span class="codeblock-lang">${label || ''}</span>
    <button class="codeblock-copy" type="button">Copy</button>
  </div>
  <pre><code class="${langAttr}" ${lineStyle}>${bodyHtml}</code></pre>
</div>`
}

marked.setOptions({
  gfm: true,
  breaks: false,
  renderer
})

export default function Markdown({ text }: { text: string }) {
  const html = useMemo(() => {
    const raw = marked.parse(text)
    const clean = DOMPurify.sanitize(raw as string, { USE_PROFILES: { html: true } })
    return clean as string
  }, [text])

  useEffect(() => {
    // wire copy buttons
    const buttons = document.querySelectorAll<HTMLButtonElement>('.codeblock .codeblock-copy')
    buttons.forEach(btn => {
      if ((btn as any).__wired) return
      ;(btn as any).__wired = true
      btn.addEventListener('click', async () => {
        const pre = btn.closest('.codeblock')?.querySelector('pre > code') as HTMLElement | null
        const raw = pre?.innerText ?? ''
        try {
          await navigator.clipboard.writeText(raw)
          const old = btn.textContent
          btn.textContent = 'Copied'
          setTimeout(() => { btn.textContent = old || 'Copy' }, 1100)
        } catch {}
      })
    })
  }, [html])

  return <div className="prose prose-invert max-w-none" dangerouslySetInnerHTML={{ __html: html }} />
}