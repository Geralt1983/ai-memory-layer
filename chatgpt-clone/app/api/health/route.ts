export const runtime = "nodejs";
import { prisma } from "@/lib/db";

export async function GET() {
  try {
    // Basic reachability
    await prisma.$queryRaw`SELECT 1`;

    // Treat "migrated" as "tables exist"
    const rows: Array<{ name: string }> = await prisma.$queryRawUnsafe(
      "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('Conversation','Message','Part')"
    );
    const migrated = Array.isArray(rows) && rows.length === 3;

    return Response.json({ ok: true, db: "up", migrated });
  } catch (e: any) {
    return new Response(JSON.stringify({ ok: false, error: e?.message ?? String(e) }), {
      status: 503,
      headers: { "Content-Type": "application/json" },
    });
  }
}