export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { conversationId, role, parts } = await req.json();
  const msg = await prisma.message.create({
    data: {
      role,
      conversation: { connect: { id: conversationId } },
      parts: { create: parts.map((p: any) => ({ type: p.type, text: p.text, code: p.code, src: p.src })) },
    },
    include: { parts: true },
  });
  await prisma.conversation.update({ where: { id: conversationId }, data: { updatedAt: new Date() } });
  return Response.json(msg);
}