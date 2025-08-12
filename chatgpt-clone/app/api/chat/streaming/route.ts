import { prisma } from '@/lib/db'

export const runtime = "nodejs"

async function* memoryEnhancedStream(content: string, messages: any[], useMemory: boolean = true, model: string = "gpt-4o-mini") {
  const encoder = new TextEncoder()
  
  if (useMemory) {
    // Try to get memory context
    try {
      const memoryResponse = await fetch('http://localhost:8001/chat/context', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: messages.map((m: any) => ({
            role: m.role,
            content: m.content
          })),
          use_memory: true,
          model
        })
      })

      if (memoryResponse.ok) {
        const memoryResult = await memoryResponse.json()
        const enhancedText = `Memory-enhanced response:\n\n${memoryResult.response}\n\n[Found ${memoryResult.memories_found} relevant memories]`
        
        for (const ch of enhancedText) {
          yield encoder.encode(ch)
          await new Promise(r => setTimeout(r, 15))
        }
        return
      }
    } catch (error) {
      console.log('Memory API not available, falling back to echo mode')
    }
  }
  
  // Fallback to echo mode
  const text = `You said: "${content}"\n\nEcho mode - memory system not available. Set OPENAI_API_KEY for real output.`
  for (const ch of text) {
    yield encoder.encode(ch)
    await new Promise(r => setTimeout(r, 10))
  }
}

export async function POST(req: Request) {
  try {
    const { chatId, content, system, model = "gpt-4o-mini" } = await req.json()
    
    // Get or create chat
    let conversation = null
    if (chatId) {
      conversation = await prisma.conversation.findUnique({
        where: { id: chatId },
        include: { messages: { include: { parts: true } } }
      })
    }
    
    if (!conversation) {
      conversation = await prisma.conversation.create({
        data: {
          name: content.slice(0, 50) || 'New chat',
          model,
          temperature: 0.7,
          systemPrompt: system
        },
        include: { messages: { include: { parts: true } } }
      })
    }

    // Save user message
    const userMessage = await prisma.message.create({
      data: {
        conversationId: conversation.id,
        role: 'user',
        parts: {
          create: {
            type: 'text',
            text: content
          }
        }
      }
    })

    // Prepare messages for streaming
    const messages = conversation.messages.map(m => ({
      role: m.role,
      content: m.parts.map(p => p.text || p.code || '').join('')
    }))
    messages.push({ role: 'user', content })

    // Start streaming response
    const stream = new ReadableStream({
      async start(controller) {
        let fullResponse = ''
        
        try {
          for await (const chunk of memoryEnhancedStream(content, messages, true, model)) {
            const text = new TextDecoder().decode(chunk)
            fullResponse += text
            controller.enqueue(chunk)
          }
        } catch (error) {
          console.error('Streaming error:', error)
        } finally {
          // Save assistant message
          await prisma.message.create({
            data: {
              conversationId: conversation!.id,
              role: 'assistant',
              parts: {
                create: {
                  type: 'text',
                  text: fullResponse
                }
              }
            }
          })
          
          controller.close()
        }
      }
    })

    return new Response(stream, {
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Transfer-Encoding': 'chunked'
      }
    })
  } catch (error: any) {
    return new Response(`Error: ${error?.message ?? String(error)}`, { status: 500 })
  }
}