import { prisma } from '@/lib/db'

export const runtime = 'nodejs'

export async function GET(_: Request, { params }: { params: { id: string } }) {
  const messages = await prisma.message.findMany({
    where: { conversationId: params.id },
    orderBy: { createdAt: 'asc' },
    include: { parts: true }
  })
  
  return Response.json({ 
    messages: messages.map(m => ({
      role: m.role,
      content: m.parts.map(p => p.text || p.code || '').join(''),
      createdAt: m.createdAt.toISOString()
    }))
  })
}

export async function DELETE(_: Request, { params }: { params: { id: string } }) {
  await prisma.conversation.delete({ where: { id: params.id } })
  return new Response(null, { status: 204 })
}

export async function PATCH(req: Request, { params }: { params: { id: string } }) {
  const { title } = await req.json()
  const chat = await prisma.conversation.update({ 
    where: { id: params.id }, 
    data: { name: title } 
  })
  return Response.json({ 
    chat: { 
      id: chat.id, 
      title: chat.name, 
      createdAt: chat.createdAt.toISOString() 
    }
  })
}