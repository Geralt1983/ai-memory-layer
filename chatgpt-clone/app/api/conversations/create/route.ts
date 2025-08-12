export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { name, model, temperature, systemPrompt } = await req.json();
  const t = await prisma.conversation.create({ data: { name, model, temperature, systemPrompt } });
  return Response.json(t);
}