export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { id } = await req.json();
  const row = await prisma.conversation.findUnique({ where: { id }, include: { messages: { include: { parts: true }, orderBy: { createdAt: "asc" } } } });
  return Response.json(row);
}