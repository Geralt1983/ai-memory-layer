export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { id, model, temperature } = await req.json();
  const t = await prisma.conversation.update({ where: { id }, data: { model, temperature } });
  return Response.json(t);
}