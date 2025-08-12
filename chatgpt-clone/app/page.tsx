import { prisma } from '@/lib/db'
import ChatUI from '@/components/ChatUI'

export default async function Home() {
  // Load initial chats
  const conversations = await prisma.conversation.findMany({
    orderBy: { updatedAt: 'desc' },
    select: { id: true, name: true, createdAt: true },
    take: 50
  })

  const initialChats = conversations.map(c => ({
    id: c.id,
    title: c.name,
    createdAt: c.createdAt.toISOString()
  }))

  return <ChatUI initialChats={initialChats} />
}