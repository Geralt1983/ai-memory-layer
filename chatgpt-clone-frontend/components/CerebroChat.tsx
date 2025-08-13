'use client'

import { useState, useEffect, useRef } from 'react'
import { Zap, Menu, Shield, Loader2 } from 'lucide-react'

interface Message {
  id: string
  content: string
  type: 'user' | 'ai'
  timestamp: string
}

interface ChatHistory {
  id: string
  title: string
  timestamp: string
  messages: Message[]
}

export default function CerebroChat() {
  const getWelcomeMessage = (): Message => ({
    id: Date.now().toString(),
    content: '🦸‍♂️ <strong>WELCOME TO THE X-MEN, MUTANT!</strong>\n\nI am Cerebro, the advanced AI system designed to assist the X-Men in their missions. I can help you with:\n\n• Mission planning and strategy\n• Mutant power analysis\n• Threat assessment\n• Team coordination\n• And much more!\n\nWhat mission shall we tackle today?',
    type: 'ai',
    timestamp: 'Just now'
  })

  const generateChatTitle = (messages: Message[]): string => {
    const userMessages = messages.filter(m => m.type === 'user')
    if (userMessages.length === 0) return 'MISSION: CLASSIFIED'
    
    const firstMessage = userMessages[0].content.toLowerCase()
    
    // Simple and effective pattern matching
    if (firstMessage.startsWith('who are') || firstMessage.includes('who are')) {
      return 'Identity Query'
    }
    if (firstMessage.startsWith('what is') || firstMessage.startsWith('what are')) {
      return 'Information Request'
    }
    if (firstMessage.startsWith('where') || firstMessage.includes('where')) {
      return 'Location Query'
    }
    if (firstMessage.startsWith('how') || firstMessage.includes('how')) {
      return 'Process Query'
    }
    if (firstMessage.startsWith('why') || firstMessage.includes('why')) {
      return 'Explanation Request'
    }
    if (firstMessage.includes('hello') || firstMessage.includes('hi ')) {
      return 'Initial Contact'
    }
    if (firstMessage.includes('help')) {
      return 'Assistance Request'
    }
    
    // Extract main topic from first meaningful word
    const words = firstMessage.split(' ').filter(w => w.length > 2)
    if (words.length > 0) {
      const mainWord = words.find(w => 
        !['the', 'and', 'are', 'can', 'you', 'what', 'who', 'where', 'when', 'how', 'why'].includes(w)
      )
      if (mainWord) {
        return mainWord.charAt(0).toUpperCase() + mainWord.slice(1) + ' Discussion'
      }
    }
    
    return 'General Chat'
  }

  // Load chat history from localStorage
  const loadChatHistory = (): ChatHistory[] => {
    if (typeof window !== 'undefined') {
      const saved = localStorage.getItem('cerebro_chat_history')
      return saved ? JSON.parse(saved) : []
    }
    return []
  }

  // Save chat history to localStorage
  const saveChatHistory = (history: ChatHistory[]) => {
    if (typeof window !== 'undefined') {
      localStorage.setItem('cerebro_chat_history', JSON.stringify(history))
    }
  }

  // Load current chat from localStorage
  const loadCurrentChat = (): { id: string, messages: Message[] } => {
    if (typeof window !== 'undefined') {
      const savedChatId = localStorage.getItem('cerebro_current_chat_id')
      const savedMessages = localStorage.getItem('cerebro_current_messages')
      if (savedChatId && savedMessages) {
        return { id: savedChatId, messages: JSON.parse(savedMessages) }
      }
    }
    const newChatId = 'chat-' + Date.now()
    return { id: newChatId, messages: [getWelcomeMessage()] }
  }

  const [currentChatId, setCurrentChatId] = useState<string>('chat-' + Date.now())
  const [messages, setMessages] = useState<Message[]>([getWelcomeMessage()])
  const [chatHistory, setChatHistory] = useState<ChatHistory[]>([])
  const [isLoaded, setIsLoaded] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const [isTyping, setIsTyping] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedCharacter, setSelectedCharacter] = useState('professor-x')
  
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }
  
  // Load data from localStorage after component mounts (client-side only)
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const loadedChat = loadCurrentChat()
      const loadedHistory = loadChatHistory()
      
      setCurrentChatId(loadedChat.id)
      setMessages(loadedChat.messages)
      setChatHistory(loadedHistory)
      setIsLoaded(true)
    }
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Save current chat to localStorage whenever messages change (only after initial load)
  useEffect(() => {
    if (isLoaded && typeof window !== 'undefined') {
      localStorage.setItem('cerebro_current_chat_id', currentChatId)
      localStorage.setItem('cerebro_current_messages', JSON.stringify(messages))
    }
  }, [currentChatId, messages, isLoaded])

  // Save chat history to localStorage whenever it changes (only after initial load)
  useEffect(() => {
    if (isLoaded) {
      saveChatHistory(chatHistory)
    }
  }, [chatHistory, isLoaded])
  
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto'
      inputRef.current.style.height = Math.min(inputRef.current.scrollHeight, 120) + 'px'
    }
  }, [inputValue])
  
  const handleSendMessage = async () => {
    if (!inputValue.trim() || isTyping) return

    // Check if this is the first user message BEFORE adding it
    const isFirstUserMessage = messages.filter(m => m.type === 'user').length === 0
    
    const newMessage: Message = {
      id: Date.now().toString(),
      content: inputValue.trim(),
      type: 'user',
      timestamp: new Date().toLocaleTimeString('en-US', { 
        hour12: false,
        hour: '2-digit',
        minute: '2-digit'
      })
    }
    
    setMessages(prev => [...prev, newMessage])
    const userInput = inputValue.trim()
    setInputValue('')
    setIsTyping(true)
    // Don't update chat history here - wait for AI response to get complete conversation
    
    try {
      // Call AI Memory Layer API
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userInput,
          thread_id: currentChatId,
          include_recent: 5,
          include_relevant: 5,
          remember_response: true
        })
      })
      
      if (!response.ok) {
        throw new Error('Failed to get response from AI')
      }
      
      const data = await response.json()
      
      // Format response with Cerebro styling
      let formattedResponse = data.response
      
      // Add Cerebro mission briefing style if it's a plain response
      if (!formattedResponse.includes('<strong>')) {
        formattedResponse = `🦸‍♂️ <strong>CEREBRO ANALYSIS COMPLETE</strong>\n\n${formattedResponse}`
      }
      
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: formattedResponse,
        type: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit'
        })
      }
      
      setMessages(prev => [...prev, aiResponse])

      // Update chat history with the complete conversation
      if (isFirstUserMessage) {
        const title = generateChatTitle([...messages, newMessage])
        
        setChatHistory(prev => {
          // Prevent duplicates
          if (prev.some(c => c.id === currentChatId)) {
            return prev
          }
          
          const newChatEntry: ChatHistory = {
            id: currentChatId,
            title,
            timestamp: aiResponse.timestamp,
            messages: [...messages, newMessage, aiResponse]
          }
          
          return [newChatEntry, ...prev]
        })
      }
    } catch (error) {
      console.error('Error calling AI Memory Layer API:', error)
      
      // Fallback response
      const errorResponse: Message = {
        id: (Date.now() + 1).toString(),
        content: "⚠️ <strong>CEREBRO CONNECTION INTERRUPTED</strong>\n\nUnable to establish connection with the main AI Memory Layer. Please check that the API is running on localhost:8000 and try again.",
        type: 'ai',
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit'
        })
      }
      
      setMessages(prev => [...prev, errorResponse])

      // Update chat history with error response if it's the first message
      if (isFirstUserMessage) {
        setChatHistory(prev => {
          const existingIndex = prev.findIndex(c => c.id === currentChatId)
          const updatedChat: ChatHistory = {
            id: currentChatId,
            title: generateChatTitle([...messages, newMessage]),
            timestamp: errorResponse.timestamp,
            messages: [...messages, newMessage, errorResponse]
          }
          
          if (existingIndex >= 0) {
            const updated = [...prev]
            updated[existingIndex] = updatedChat
            return updated
          } else {
            return [updatedChat, ...prev]
          }
        })
      }
    } finally {
      setIsTyping(false)
    }
  }
  
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }
  
  const createNewChat = () => {
    // Save current chat to history if it has messages beyond the welcome message
    if (messages.length > 1) {
      const title = generateChatTitle(messages)
      
      const newChatEntry: ChatHistory = {
        id: currentChatId,
        title,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit'
        }),
        messages: [...messages]
      }
      
      setChatHistory(prev => {
        // Prevent duplicates
        if (prev.some(c => c.id === currentChatId)) {
          return prev
        }
        return [newChatEntry, ...prev]
      })
    }
    
    // Create new chat
    const newChatId = 'chat-' + Date.now()
    setCurrentChatId(newChatId)
    setMessages([getWelcomeMessage()])
    setSidebarOpen(false)
  }
  
  const loadChat = (chat: ChatHistory) => {
    // Save current chat first if needed
    if (messages.length > 1 && currentChatId !== chat.id) {
      const title = generateChatTitle(messages)
        
      const currentChat: ChatHistory = {
        id: currentChatId,
        title,
        timestamp: new Date().toLocaleTimeString('en-US', { 
          hour12: false,
          hour: '2-digit',
          minute: '2-digit'
        }),
        messages: [...messages]
      }
      
      // Update or add current chat to history
      setChatHistory(prev => {
        const existingIndex = prev.findIndex(c => c.id === currentChatId)
        if (existingIndex >= 0) {
          const updated = [...prev]
          updated[existingIndex] = currentChat
          return updated
        }
        return [currentChat, ...prev]
      })
    }
    
    // Load selected chat
    setCurrentChatId(chat.id)
    setMessages(chat.messages)
    setSidebarOpen(false)
  }
  
  const toggleTheme = () => {
    // Clear localStorage for testing
    if (typeof window !== 'undefined') {
      localStorage.removeItem('cerebro_chat_history')
      localStorage.removeItem('cerebro_current_chat_id')
      localStorage.removeItem('cerebro_current_messages')
      setChatHistory([])
      setMessages([getWelcomeMessage()])
      setCurrentChatId('chat-' + Date.now())
    }
  }
  
  return (
    <div className="flex h-screen backdrop-blur-sm">
      {/* Sidebar */}
      <aside className={`cerebro-panel flex flex-col overflow-hidden w-80 flex-shrink-0 ${sidebarOpen ? 'fixed left-0 top-0 h-screen z-50 md:relative' : 'hidden md:flex'}`}>
        {/* Sidebar Header - Sticky */}
        <div className="sidebar-header flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="x-logo">X</div>
            <h1 className="header-title">CEREBRO</h1>
          </div>
          <div className="status-indicator">
            <div className="status-dot"></div>
            <span>MUTANT DETECTED</span>
          </div>
        </div>
        
        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto">
          <div className="p-6">
            <button 
              className="btn btn-primary w-full mb-6" 
              onClick={createNewChat}
            >
              <Zap className="w-4 h-4 mr-2" />
              NEW MISSION
            </button>
            
            <div className="mb-4">
              <div className="text-sm text-muted mb-2" style={{fontFamily: 'var(--font-apocalypse)'}}>
                MUTANT POWER LEVEL
              </div>
              <div className="power-meter">
                <div className="power-fill"></div>
              </div>
            </div>
            
            <div className="space-y-3" style={{fontFamily: 'var(--font-apocalypse)'}}>
              {chatHistory.length === 0 ? (
                <div className="text-center text-muted text-sm py-4">
                  No previous missions
                </div>
              ) : (
                chatHistory.map(chat => (
                  <div 
                    key={chat.id}
                    className={`p-3 rounded-lg cursor-pointer border-2 transition-colors ${
                      currentChatId === chat.id 
                        ? 'border-accent bg-tertiary' 
                        : 'border-x-men-gold hover:border-accent'
                    }`}
                    style={{fontFamily: 'var(--font-apocalypse)'}}
                    onClick={() => loadChat(chat)}
                    title={chat.title} // Add tooltip for full title
                  >
                    <div className="font-bold text-sm text-x-men-gold leading-tight break-words overflow-hidden"
                         style={{
                           display: '-webkit-box',
                           WebkitLineClamp: 2,
                           WebkitBoxOrient: 'vertical'
                         }}>
                      {chat.title}
                    </div>
                    <div className="text-xs text-muted mt-1">{chat.timestamp}</div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
        
        {/* Sidebar Footer - Sticky */}
        <div className="flex-shrink-0 p-6 border-t border-x-men-gold">
          <button 
            className="btn w-full justify-start" 
            onClick={toggleTheme}
          >
            <Shield className="w-4 h-4 mr-2" />
            TOGGLE POWERS
          </button>
        </div>
      </aside>
      
      {/* Main Content */}
      <main className="flex flex-col flex-1 min-w-0">
        {/* Header */}
        <header className="header">
          <div className="flex items-center gap-4">
            <button 
              className="btn p-2 md:hidden" 
              onClick={() => setSidebarOpen(!sidebarOpen)}
            >
              <Menu className="w-5 h-5" />
            </button>
            <h2 className="header-title">X-MEN MISSION CONTROL</h2>
          </div>
          
          <div className="flex items-center gap-4">
            <select 
              className="bg-tertiary border-2 border-x-men-gold rounded-lg px-3 py-2 text-sm"
              style={{fontFamily: 'var(--font-apocalypse)'}}
              value={selectedCharacter}
              onChange={(e) => setSelectedCharacter(e.target.value)}
            >
              <option value="professor-x">PROFESSOR X</option>
              <option value="cyclops">CYCLOPS</option>
              <option value="wolverine">WOLVERINE</option>
              <option value="storm">STORM</option>
            </select>
            <div className="status-indicator">
              <div className="status-dot"></div>
              <span>MISSION ACTIVE</span>
            </div>
          </div>
        </header>
        
        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.map(message => (
              <div 
                key={message.id}
                className={`message ${message.type === 'user' ? 'message-user' : 'message-ai cerebro-panel'}`}
              >
                <div 
                  className="message-content"
                  dangerouslySetInnerHTML={{ __html: message.content }}
                />
                <div className="message-timestamp">
                  [{message.type === 'user' ? 'MUTANT' : 'CEREBRO'}] {message.timestamp}
                </div>
              </div>
            ))}
            
            {/* Typing Indicator */}
            {isTyping && (
              <div className="message message-ai cerebro-panel">
                <div className="message-content">
                  <div className="typing-indicator">
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <div className="typing-dot"></div>
                    <span style={{marginLeft: '12px', color: 'var(--x-men-gold)'}}>ANALYZING...</span>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>
        </div>
        
        {/* Input Area */}
        <div className="input-container">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-4">
              <textarea 
                ref={inputRef}
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={handleKeyDown}
                className="chat-input flex-1"
                placeholder="Enter your mission briefing..."
                rows={2}
              />
              <button 
                className="btn btn-primary px-6" 
                onClick={handleSendMessage}
                disabled={isTyping}
              >
                {isTyping ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Zap className="w-5 h-5" />
                )}
              </button>
            </div>
            <div className="mt-3 text-xs text-muted flex justify-between" style={{fontFamily: 'var(--font-apocalypse)'}}>
              <span>Press ENTER to transmit | SHIFT+ENTER for new line</span>
              <span>MUTANT STATUS: ACTIVE</span>
            </div>
          </div>
        </div>
      </main>
      
      {/* Mobile overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}
    </div>
  )
}
