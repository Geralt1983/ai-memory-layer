'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import type { UIMessage } from '@/lib/types'
import Markdown from './Markdown'

interface ChatRef { id: string; title: string; createdAt: string }

export default function ChatUI({ initialChats }: { initialChats: ChatRef[] }) {
  const [chats, setChats] = useState(initialChats)
  const [activeId, setActiveId] = useState<string | null>(chats[0]?.id ?? null)
  const [messages, setMessages] = useState<UIMessage[]>([])
  const [input, setInput] = useState('')
  const [system, setSystem] = useState('You are a helpful personal assistant.')
  const [model, setModel] = useState<string>(process.env.NEXT_PUBLIC_OPENAI_MODEL || 'gpt-4o-mini')
  const [streaming, setStreaming] = useState(false)
  const [theme, setTheme] = useState<'light'|'dark'>(() =>
    typeof document !== 'undefined' && document.documentElement.classList.contains('dark') ? 'dark' : 'light'
  )
  const [editingIndex, setEditingIndex] = useState<number | null>(null)
  const [mobileSidebarOpen, setMobileSidebarOpen] = useState(false)
  const [userScrolled, setUserScrolled] = useState(false)

  const listRef = useRef<HTMLDivElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const abortRef = useRef<AbortController | null>(null)

  // Theme sync
  useEffect(() => {
    const isDark = document.documentElement.classList.contains('dark')
    setTheme(isDark ? 'dark' : 'light')
  }, [])

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current
    if (!textarea) return
    textarea.style.height = 'auto'
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px'
  }, [input])

  function toggleTheme() {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    const el = document.documentElement
    if (next === 'dark') el.classList.add('dark'); else el.classList.remove('dark')
    el.style.colorScheme = next
    try { localStorage.setItem('theme', next) } catch {}
  }

  // Load messages when active chat changes
  useEffect(() => {
    if (!activeId) return
    ;(async () => {
      const res = await fetch(`/api/chats/${activeId}`)
      const data = await res.json()
      setMessages(data.messages)
    })()
  }, [activeId])

  // Auto scroll to bottom, but track user scroll
  useEffect(() => {
    if (!listRef.current) return
    
    const handleScroll = () => {
      if (!listRef.current) return
      const { scrollTop, scrollHeight, clientHeight } = listRef.current
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50
      setUserScrolled(!isAtBottom)
    }
    
    const container = listRef.current
    container.addEventListener('scroll', handleScroll)
    
    // Auto-scroll if not manually scrolled or if streaming just started
    if (!userScrolled || streaming) {
      container.scrollTop = container.scrollHeight
    }
    
    return () => container.removeEventListener('scroll', handleScroll)
  }, [messages, streaming, userScrolled])

  async function createNewChat() {
    const res = await fetch('/api/chats', { method: 'POST' })
    const data = await res.json()
    setChats(prev => [data.chat, ...prev])
    setActiveId(data.chat.id)
    setMessages([])
    setMobileSidebarOpen(false)
  }

  async function deleteChat(id: string) {
    await fetch(`/api/chats/${id}`, { method: 'DELETE' })
    setChats(prev => prev.filter(c => c.id !== id))
    if (activeId === id) {
      setActiveId(chats.find(c => c.id !== id)?.id ?? null)
      setMessages([])
    }
  }

  async function renameActive() {
    if (!activeId) return
    const current = chats.find(c => c.id === activeId)?.title || 'Untitled'
    const title = prompt('Rename chat', current)
    if (!title) return
    await fetch(`/api/chats/${activeId}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ title }) })
    const list = await fetch('/api/chats').then(r => r.json())
    setChats(list.chats)
  }

  async function refreshChats() {
    const list = await fetch('/api/chats').then(r => r.json())
    setChats(list.chats)
  }

  async function sendMessage(e?: React.FormEvent) {
    e?.preventDefault()
    if (streaming) return
    const content = editingIndex !== null ? messages[editingIndex].content : input
    if (!content?.trim()) return

    // Focus management & scroll reset
    setUserScrolled(false)
    textareaRef.current?.focus()

    // Prepare optimistic UI
    let payloadChatId = activeId
    let optimistic: UIMessage[] = []
    if (editingIndex !== null) {
      const before = messages.slice(0, editingIndex)
      optimistic = [...before, { role: 'user', content }, { role: 'assistant', content: '' }]
      setMessages(optimistic)
      setEditingIndex(null)
    } else {
      setInput('')
      optimistic = [...messages, { role: 'user', content }, { role: 'assistant', content: '' }]
      setMessages(optimistic)
    }

    setStreaming(true)
    abortRef.current = new AbortController()

    try {
      // Use unified streaming endpoint
      const res = await fetch('/api/chat/streaming', {
        method: 'POST',
        signal: abortRef.current.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chatId: payloadChatId, content, system, model })
      })

      if (!res.body) throw new Error('No response body')
      
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let acc = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break
        acc += decoder.decode(value, { stream: true })
        setMessages(m => {
          const copy = m.slice()
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') last.content = acc
          return copy
        })
      }
    } catch (error: any) {
      if (error.name !== 'AbortError') {
        console.error('Streaming error:', error)
        setMessages(m => {
          const copy = m.slice()
          const last = copy[copy.length - 1]
          if (last?.role === 'assistant') {
            last.content = 'Sorry, there was an error processing your request.'
          }
          return copy
        })
      }
    } finally {
      setStreaming(false)
      await refreshChats()
      if (!activeId) {
        const list = await fetch('/api/chats').then(r => r.json())
        setActiveId(list.chats[0]?.id ?? null)
      }
    }
  }

  function stopStreaming() {
    try { abortRef.current?.abort() } catch {}
    setStreaming(false)
  }

  function regenerate() {
    if (streaming || messages.length < 2) return
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') { 
        setEditingIndex(i)
        setInput(messages[i].content)
        break 
      }
    }
  }

  function scrollToBottom() {
    if (!listRef.current) return
    listRef.current.scrollTop = listRef.current.scrollHeight
    setUserScrolled(false)
  }

  // Handle Enter vs Shift+Enter
  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const title = useMemo(() => chats.find(c => c.id === activeId)?.title ?? 'New chat', [activeId, chats])

  // Keyboard shortcuts
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const isMac = navigator.platform.toLowerCase().includes('mac')
      const meta = isMac ? e.metaKey : e.ctrlKey
      if (meta && e.key.toLowerCase() === 'enter') { e.preventDefault(); sendMessage() }
      if (e.key === 'Escape' && streaming) { e.preventDefault(); stopStreaming() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [streaming, messages])

  const sidebarContent = (
    <aside className="border-r border-white/10 bg-card-light dark:bg-card-dark p-3 flex flex-col gap-3 h-full">
      <button 
        onClick={createNewChat} 
        className="w-full rounded-2xl bg-emerald-600 hover:bg-emerald-500 text-white py-2 text-sm shadow"
        aria-label="Create new chat"
      >
        + New chat
      </button>
      <div className="flex-1 overflow-auto space-y-1">
        {chats.map(c => (
          <div 
            key={c.id} 
            className={`group flex items-center justify-between gap-2 px-3 py-2 rounded-xl cursor-pointer ${
              activeId === c.id ? 'bg-white/60 dark:bg-white/10' : 'hover:bg-white/50 dark:hover:bg-white/10'
            }`}
            onClick={() => {setActiveId(c.id); setMobileSidebarOpen(false)}}
            role="button"
            tabIndex={0}
            aria-label={`Switch to chat: ${c.title}`}
          >
            <span className="truncate text-sm">{c.title}</span>
            <div className="opacity-0 group-hover:opacity-100 flex items-center gap-2 text-xs">
              <button 
                onClick={(e) => {e.stopPropagation(); setActiveId(c.id); renameActive()}} 
                className="text-gray-400 hover:text-gray-200"
                aria-label="Rename chat"
              >
                Rename
              </button>
              <button 
                onClick={(e) => {e.stopPropagation(); deleteChat(c.id)}} 
                className="text-red-400 hover:text-red-300"
                aria-label="Delete chat"
              >
                Delete
              </button>
            </div>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-between text-xs opacity-80">
        <button 
          onClick={toggleTheme} 
          className="px-2 py-1 rounded-lg bg-white/60 dark:bg-white/10"
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} theme`}
        >
          <span suppressHydrationWarning>{theme === 'dark' ? 'Light' : 'Dark'}</span>
        </button>
        <a className="underline" href="https://platform.openai.com/" target="_blank" rel="noreferrer">
          OpenAI
        </a>
      </div>
    </aside>
  )

  return (
    <div className="grid h-screen grid-cols-1 md:grid-cols-[280px_1fr] relative">
      {/* Mobile sidebar overlay */}
      {mobileSidebarOpen && (
        <div className="fixed inset-0 bg-black/50 z-40 md:hidden" onClick={() => setMobileSidebarOpen(false)} />
      )}
      
      {/* Sidebar - responsive */}
      <div className={`
        fixed md:relative top-0 left-0 h-full w-[280px] z-50 md:z-auto
        transform transition-transform duration-200 ease-in-out md:transform-none
        ${mobileSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
      `}>
        {sidebarContent}
      </div>

      {/* Main */}
      <main className="flex flex-col h-screen">
        {/* Header */}
        <header className="border-b border-white/10 px-4 md:px-6 py-3 flex items-center gap-3 bg-bg-light/60 dark:bg-bg-dark/60 sticky top-0 backdrop-blur">
          <button
            onClick={() => setMobileSidebarOpen(true)}
            className="md:hidden p-1 rounded-lg bg-white/60 dark:bg-white/10"
            aria-label="Open sidebar"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <h1 className="text-lg font-semibold truncate flex items-center gap-2">
            <span>{title}</span>
            <span className="text-xs px-2 py-0.5 rounded-lg bg-white/10 border border-white/10">{model}</span>
            <span className="px-2 py-0.5 rounded bg-green-500 text-white text-xs font-bold">LIVE</span>
          </h1>
          <div className="ml-auto flex items-center gap-2 text-xs">
            <select 
              value={model} 
              onChange={e => setModel(e.target.value)} 
              className="rounded-lg bg-white/60 dark:bg-white/10 px-2 py-1"
              aria-label="Select AI model"
            >
              <option value="gpt-4o-mini">gpt-4o-mini</option>
              <option value="gpt-4o">gpt-4o</option>
            </select>
            <details>
              <summary className="cursor-pointer px-2 py-1 rounded-lg bg-white/60 dark:bg-white/10">
                Settings
              </summary>
              <div className="absolute right-4 mt-2 w-[280px] md:w-[480px] rounded-2xl border border-white/10 bg-card-light dark:bg-card-dark p-3 shadow-xl">
                <label className="text-xs opacity-70">System prompt</label>
                <textarea 
                  value={system} 
                  onChange={e => setSystem(e.target.value)} 
                  className="w-full mt-1 h-32 rounded-xl bg-white/60 dark:bg-white/10 p-2"
                  aria-label="System prompt"
                />
              </div>
            </details>
            <button 
              onClick={stopStreaming} 
              disabled={!streaming} 
              className="hidden sm:block rounded-lg px-2 py-1 bg-red-600/80 text-white disabled:opacity-50"
              aria-label="Stop generation"
            >
              Stop
            </button>
            <button 
              onClick={regenerate} 
              disabled={streaming || messages.length < 2} 
              className="hidden sm:block rounded-lg px-2 py-1 bg-white/60 dark:bg-white/10 disabled:opacity-50"
              aria-label="Regenerate last response"
            >
              Regenerate
            </button>
          </div>
        </header>

        {/* Messages */}
        <div ref={listRef} className="flex-1 overflow-auto px-4 md:px-6">
          <div className="mx-auto max-w-3xl py-6 space-y-6">
            {messages.map((m, i) => (
              <div 
                key={i} 
                className={`rounded-2xl p-4 ${
                  m.role === 'user' ? 'bg-white/70 dark:bg-white/10' : 'bg-white/5 border border-white/10'
                }`}
              >
                {m.role === 'assistant' ? (
                  <Markdown text={m.content} />
                ) : (
                  <div className="whitespace-pre-wrap">{m.content}</div>
                )}
                {m.role === 'user' && (() => {
                  for (let j = messages.length - 1; j >= 0; j--) {
                    if (messages[j].role === 'user') return i === j;
                  }
                  return false;
                })() && (
                  <div className="mt-2">
                    <button 
                      className="text-xs opacity-80 underline hover:opacity-100" 
                      onClick={() => { setEditingIndex(i); setInput(m.content); }}
                      aria-label="Edit and regenerate this message"
                    >
                      Edit and regenerate
                    </button>
                  </div>
                )}
              </div>
            ))}
            {streaming && (
              <div className="animate-pulse text-sm opacity-80 flex items-center gap-2">
                Assistant is typing…
                <button 
                  onClick={stopStreaming}
                  className="sm:hidden text-xs bg-red-600/80 text-white px-2 py-1 rounded"
                  aria-label="Stop generation"
                >
                  Stop
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Jump to latest button */}
        {userScrolled && streaming && (
          <div className="absolute bottom-24 right-6 z-10">
            <button
              onClick={scrollToBottom}
              className="bg-emerald-600 hover:bg-emerald-500 text-white px-3 py-2 rounded-full shadow-lg text-sm"
              aria-label="Jump to latest message"
            >
              ↓ Jump to latest
            </button>
          </div>
        )}

        {/* Composer */}
        <form onSubmit={sendMessage} className="border-t border-white/10 p-4">
          <div className="mx-auto max-w-3xl flex gap-2">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message..."
              className="flex-1 rounded-2xl bg-white/70 dark:bg-white/10 px-4 py-3 resize-none overflow-hidden min-h-[48px]"
              rows={1}
              aria-label="Type your message"
            />
            <button 
              type="submit"
              disabled={streaming || !input.trim()} 
              className="rounded-2xl px-4 py-3 bg-emerald-600 hover:bg-emerald-500 text-white disabled:opacity-50 self-end"
              aria-label="Send message"
            >
              Send
            </button>
          </div>
          <div className="mx-auto max-w-3xl mt-2 text-xs opacity-70">
            <span className="hidden sm:inline">Cmd+Enter to send • </span>
            Enter to send, Shift+Enter for new line
            <span className="hidden sm:inline"> • Esc to stop</span>
          </div>
        </form>
      </main>
    </div>
  )
}