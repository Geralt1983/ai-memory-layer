export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function GET() {
  const rows = await prisma.conversation.findMany({ orderBy: { updatedAt: "desc" }, select: { id: true, name: true, model: true, temperature: true, systemPrompt: true, updatedAt: true, createdAt: true } });
  return Response.json(rows);
}