import { prisma } from '@/lib/db'

export const runtime = 'nodejs'

export async function GET() {
  const chats = await prisma.conversation.findMany({
    orderBy: { updatedAt: 'desc' },
    select: { id: true, name: true, createdAt: true }
  })
  return Response.json({ 
    chats: chats.map(c => ({ 
      id: c.id, 
      title: c.name, 
      createdAt: c.createdAt.toISOString() 
    }))
  })
}

export async function POST() {
  const chat = await prisma.conversation.create({
    data: {
      name: 'New chat',
      model: 'gpt-4o-mini',
      temperature: 0.7
    }
  })
  return Response.json({ 
    chat: { 
      id: chat.id, 
      title: chat.name, 
      createdAt: chat.createdAt.toISOString() 
    }
  })
}