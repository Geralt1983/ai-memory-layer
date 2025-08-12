export const runtime = "nodejs";
import { prisma } from "@/lib/db";
export async function POST(req: Request) {
  const { id, name } = await req.json();
  const t = await prisma.conversation.update({ where: { id }, data: { name } });
  return Response.json(t);
}