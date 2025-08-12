import { prisma } from '@/lib/db'

export const runtime = 'nodejs'

export async function POST(req: Request) {
  try {
    const { chatId, content, system, model = "gpt-4o-mini", useMemory = true } = await req.json()
    
    // Get or create conversation
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
    await prisma.message.create({
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

    // Prepare messages for memory context
    const messages = conversation.messages.map(m => ({
      role: m.role,
      content: m.parts.map(p => p.text || p.code || '').join('')
    }))
    messages.push({ role: 'user', content })

    // Try memory-enhanced response with improved error handling
    let responseText = ''
    if (useMemory) {
      try {
        // Test CORS with OPTIONS first (matches backend improvements)
        await fetch('http://localhost:8001/chat/context', { method: 'OPTIONS' })
        
        const memoryResponse = await fetch('http://localhost:8001/chat/context', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages,
            use_memory: true,
            model
          })
        })

        if (memoryResponse.ok) {
          const memoryResult = await memoryResponse.json()
          const memoriesInfo = memoryResult.memories_found > 0 
            ? `\n\n*[Enhanced with ${memoryResult.memories_found} relevant memories]*`
            : '\n\n*[Memory system active]*'
          responseText = `${memoryResult.response}${memoriesInfo}`
        } else {
          console.log(`Memory API returned ${memoryResponse.status}`)
        }
      } catch (error) {
        console.log('Memory API not available:', error instanceof Error ? error.message : 'Unknown error')
      }
    }
    
    // Fallback response
    if (!responseText) {
      responseText = `You said: "${content}"\n\nEcho mode - Set OPENAI_API_KEY for real responses or start the memory API for context-aware responses.`
    }

    // Stream the response
    const encoder = new TextEncoder()
    const stream = new ReadableStream({
      async start(controller) {
        // Simulate streaming by sending character by character
        for (const char of responseText) {
          controller.enqueue(encoder.encode(char))
          await new Promise(resolve => setTimeout(resolve, 15))
        }
        
        // Save assistant message
        await prisma.message.create({
          data: {
            conversationId: conversation!.id,
            role: 'assistant',
            parts: {
              create: {
                type: 'text',
                text: responseText
              }
            }
          }
        })
        
        controller.close()
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