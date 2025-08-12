export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { id, systemPrompt } = await req.json();
  const t = await prisma.conversation.update({ where: { id }, data: { systemPrompt } });
  return Response.json(t);
}