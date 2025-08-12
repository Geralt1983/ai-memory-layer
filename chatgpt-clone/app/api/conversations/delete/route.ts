export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { id } = await req.json();
  await prisma.part.deleteMany({ where: { message: { conversationId: id } } });
  await prisma.message.deleteMany({ where: { conversationId: id } });
  await prisma.conversation.delete({ where: { id } });
  return Response.json({ ok: true });
}